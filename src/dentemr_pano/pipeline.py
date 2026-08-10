"""Layers 1-3 of the translation pipeline: translate, back-translate, check.

Layer 1 -- forward translation of the deduplicated segment table, at
``temperature=0``, with the manuscript's abbreviation table as glossary. Two
models translate every segment independently; where they disagree the segment
is flagged rather than silently resolved.

Layer 2 -- independent back-translation. The English is rendered back into
Chinese by *the other* model, and the round trip is scored by cosine similarity
of sentence embeddings. Using the same model for both directions would measure
its self-consistency, not the translation's fidelity, so the assignment is
deliberately crossed.

Layer 3 -- deterministic rule checks that need no model: protected-token
preservation, negation alignment, and length inflation.

Segments are sent in batches. Every model here is a reasoning model, so
per-call latency is dominated by a fixed thinking cost; batching amortises it
and is the difference between a twenty-minute run and a six-hour one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .segments import missing_tokens

# Bare FDI tooth numbers -- "24", "36" without a leading "#". These are not in
# PROTECTED_PATTERNS: masking every two-digit number would swallow durations
# ("3 days"), counts and dates, and the sentinels would outnumber the words.
# They are therefore left in the text for the model to copy through, and
# checked here instead. Quadrant 1-4, position 1-8, not part of a longer number
# and not preceded by a decimal point.
# The "#" is optional on purpose. Chinese records write tooth numbers bare;
# the translation often adds the hash ("24、25" -> "#24, #25"). Excluding
# hashed forms counted the tooth on the Chinese side only and reported a
# phantom loss on every such segment.
# The lookbehind excludes a preceding digit but not a preceding "." -- in this
# corpus the full stop is overloaded, serving as decimal point, list separator
# and an occasional typed substitution for 、. Excluding it dropped genuine teeth ("2.23行
# 根管治疗" is item 2, tooth 23). Decimal measurements are already masked as
# sentinels by layer 0, so the ambiguity is largely resolved before this runs.
_BARE_TOOTH = re.compile(r"(?<![0-9])#?\s?([1-4][1-8])(?![0-9])")
#: Leading list enumerators -- "12、", "2. ", "3)" -- are item numbers, not
#: teeth. They must be stripped before matching or every numbered treatment
#: plan reports a phantom tooth.
#: A list enumerator and the opening of a tooth list are written identically in
#: Chinese: "12、阿替卡因" is item 12, "34、37牙" is teeth 34 and 37. They are
#: told apart by what follows the separator -- another FDI-range number means
#: it is a tooth list, not an enumerator.
_ENUMERATOR = re.compile(
    r"(?:^|(?<=[\n；;。]))\s*\d{1,2}\s*[、.)．）]\s*(?![1-4][1-8](?![0-9]))"
)


def bare_teeth(text: str) -> set[str]:
    """FDI tooth numbers written without a leading ``#``.

    Returns a *set*, not a multiset: Chinese clinical prose names the same
    tooth more than once far more readily than English does, so comparing
    counts flags ordinary rewording. What matters is that no tooth appears on
    one side and not the other.

    This is a heuristic. Two-digit numbers in the FDI range also occur as ages,
    durations and quantities, so it can produce false positives -- but only
    asymmetric ones are reported, and a spurious match usually appears on both
    sides and cancels.
    """
    t = re.sub(r"⟦P\d+⟧", "", text or "")
    t = _ENUMERATOR.sub(" ", t)
    return set(_BARE_TOOTH.findall(t))

#: Negation markers that must survive the round trip. A translation that drops
#: a negation inverts the clinical meaning, which no similarity score reliably
#: catches -- "no percussion pain" and "percussion pain" embed close together.
# Bare "不" cannot be a negation marker on its own: 不适 (discomfort), 不良
# (poor), 不全 (incomplete) are nouns and adjectives that carry no negation of
# the clause. Matching it naively flagged every "不适随诊" (return if discomfort)
# as a lost negation. The pattern below requires 不 to negate a verb.
_ZH_NEG_RE = __import__("re").compile(
    r"无(?!菌|痛性)|未(?!成年)|否认|阴性|禁用|禁止|勿|\(-\)|（-）|\(－\)|"
    # 不 negates a clause only outside these compounds. 牙列不齐 (irregular
    # dentition), 固位不佳 (poor retention) and 钙化不通 (calcified, not
    # patent) are descriptive terms, not negations of the sentence.
    r"不(?!适|良|全|等|同|规则|均匀|明显|齐|佳|清|详)"
)
_EN_NEG = (" no ", " not ", " without ", " absent", " negative", " denies",
           " denied", " denying", " free of ", "(-)", " non-", " none",
           " nil", " nothing", " unremarkable", " n/a", " neither", " never",
           " lack", " no.", " cannot", " can not", " unable", " could not",
           " failed to", " less than", " avoid", " refrain", " prohibit")

_FWD_SYSTEM = """\
You are a bilingual dental clinician translating Chinese outpatient dental \
record fragments into English for a public research dataset.

Rules:
1. Translate meaning, not words. Produce natural clinical English.
2. Sentinels of the form ⟦P0⟧, ⟦P1⟧ are protected tokens (tooth numbers, ICD \
codes, drug strings, measurements). Reproduce every sentinel EXACTLY as it \
appears, in the position the English requires. Never renumber, translate, \
expand or drop a sentinel.
3. Expand clinical abbreviations to the English full term using the glossary.
4. Preserve negation and hedging exactly. 无/未/不/(-) must map to an explicit \
English negation. 考虑 is "consider", not a definite diagnosis.
5. Add nothing, omit nothing, explain nothing.

You will receive a JSON object mapping ids to Chinese fragments. Return STRICT \
JSON with the same ids mapping to English strings. No markdown fence.
"""

_BACK_SYSTEM = """\
You translate English dental record fragments back into Chinese.

Rules:
1. Translate faithfully and literally. Do NOT improve, expand or clean up the \
English. The output is used to check the English against an original, so any \
embellishment corrupts the measurement.
2. Sentinels of the form ⟦P0⟧ are opaque tokens. Reproduce them exactly.
3. Preserve negation exactly.

You will receive a JSON object mapping ids to English fragments. Return STRICT \
JSON with the same ids mapping to Chinese strings. No markdown fence.
"""


def _glossary(gloss: dict[str, str]) -> str:
    return "\n".join(f"{k} = {v}" for k, v in gloss.items())


def forward_system(gloss: dict[str, str]) -> str:
    return _FWD_SYSTEM + "\nGlossary:\n" + _glossary(gloss)


def back_system() -> str:
    return _BACK_SYSTEM


def batch(items: list[str], size: int) -> list[list[tuple[str, str]]]:
    """Chunk segments into id/text batches; ids are the index within the batch."""
    out, cur = [], []
    for i, s in enumerate(items):
        cur.append((str(len(cur)), s))
        if len(cur) == size:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def parse_batch(raw: str, expect: list[tuple[str, str]]) -> dict[str, str]:
    """Parse a batch response, tolerating a stray markdown fence."""
    txt = raw.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```[a-z]*\n|\n```$", "", txt)
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict):
        return {}
    return {k: v for k, v in obj.items() if isinstance(v, str)}


# -- Layer 3 ---------------------------------------------------------------

@dataclass
class RuleReport:
    token_ok: bool
    negation_ok: bool
    length_ok: bool
    dropped_tokens: list[str]
    tooth_ok: bool = True
    tooth_detail: str = ""
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.token_ok and self.negation_ok and self.length_ok and self.tooth_ok


def has_zh_negation(s: str) -> bool:
    return bool(_ZH_NEG_RE.search(s))


def has_en_negation(s: str) -> bool:
    low = f" {s.lower()} "
    return any(m in low for m in _EN_NEG)


def normalise_en(s: str) -> str:
    """Casefold, strip punctuation and collapse whitespace.

    Used to decide whether two models genuinely disagree. Raw string equality
    calls "Digital panoramic radiograph" and "digital panoramic radiograph" a
    disagreement, which would inflate the reported divergence rate with pure
    typography and bury the cases where the models actually chose different
    clinical wording.
    """
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", s.lower()).strip()


def check_rules(zh: str, en: str, tokens: list[str],
                max_ratio: float = 9.0, min_ratio: float = 0.8) -> RuleReport:
    """Deterministic checks on one translated segment.

    Length is compared as English characters per Chinese character. Chinese is
    far denser than English -- a four-character diagnosis routinely becomes a
    thirty-character English phrase -- so the expected ratio sits well above 1
    and the upper bound has to be generous. The bounds are set to catch a
    translation that collapsed to a stub or ballooned into commentary, not
    ordinary expansion; an empirical sweep over this corpus put almost all
    correct translations between 1.5 and 7.
    """
    dropped = missing_tokens(en, tokens)
    token_ok = not dropped

    zh_teeth, en_teeth = bare_teeth(zh), bare_teeth(en)
    tooth_ok = zh_teeth == en_teeth
    tooth_detail = "" if tooth_ok else (
        f"bare teeth missing {sorted(zh_teeth - en_teeth)} "
        f"added {sorted(en_teeth - zh_teeth)}")

    zh_neg, en_neg = has_zh_negation(zh), has_en_negation(en)
    negation_ok = zh_neg == en_neg

    zh_len = max(len(re.sub(r"⟦P\d+⟧", "", zh)), 1)
    en_len = len(re.sub(r"⟦P\d+⟧", "", en))
    ratio = en_len / zh_len
    # Very short sources have wildly variable legitimate ratios -- 无 -> "None"
    # is 4.0 -- so the ratio test is relaxed for them. It is not waived: an
    # absolute cap still catches a two-character complaint that came back as a
    # paragraph of invented commentary.
    if zh_len < 4:
        length_ok = en_len <= 40
    else:
        length_ok = min_ratio <= ratio <= max_ratio

    bits = []
    if not token_ok:
        bits.append(f"dropped {dropped}")
    if not negation_ok:
        bits.append(f"negation zh={zh_neg} en={en_neg}")
    if not length_ok:
        bits.append(f"len ratio {ratio:.2f}")
    if not tooth_ok:
        bits.append(tooth_detail)
    return RuleReport(token_ok, negation_ok, length_ok, dropped,
                      tooth_ok, tooth_detail, "; ".join(bits))
