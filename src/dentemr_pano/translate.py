"""Chinese-to-English translation of the clinical free-text fields.

The v2 archive ships bilingual: every Chinese narrative field keeps its original
value and gains an ``_en`` sibling. Translation is machine-produced with a fixed
model at temperature 0 and then sampled for clinician review, so both halves of
the process are reproducible from this module.

Design constraints that the prompt enforces:

* FDI tooth notation (``#36``, ``36``, ``11``) is a code, not a word. It is
  copied through untouched; renumbering or "correcting" it corrupts the record.
* Measurements, ICD-10 codes and drug dosages are copied verbatim.
* Clinical abbreviations are expanded to their English full term using the
  glossary below, which is the same table printed in the manuscript. This is
  the one place translation deliberately departs from source fidelity: the
  Chinese field retains the abbreviation, the English field spells it out.
* An empty source field translates to an empty string, never to prose.

Nothing here is specific to the source hospital, so this module is published in
full (see ``docs/EXCLUSIONS.md``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

#: Providers, each an OpenAI-compatible chat-completions endpoint. Both are
#: used deliberately: the forward translation and the back-translation must not
#: come from the same vendor, or the round trip measures one vendor's
#: self-consistency instead of the translation's fidelity.
PROVIDERS: dict[str, tuple[str, str]] = {
    # name: (environment variable holding the key, endpoint)
    "deepseek": ("deepseek_api", "https://api.deepseek.com/chat/completions"),
    "qwen": ("qwen_api",
             "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
}

API_URL = PROVIDERS["deepseek"][1]
DEFAULT_MODEL = "deepseek-v4-flash"

#: The eight Chinese free-text fields that receive an ``_en`` sibling.
TRANSLATABLE_FIELDS: tuple[str, ...] = (
    "chief_complaint",
    "history_of_present_illness",
    "oral_examination",
    "imaging_examination",
    "primary_diagnosis",
    "treatment_plan",
    "procedure",
    "physician_advice",
)

#: Abbreviation glossary, identical to the manuscript's normalization table.
GLOSSARY: dict[str, str] = {
    "PD": "probing depth",
    "AL": "attachment loss",
    "BOP": "bleeding on probing",
    "GR": "gingival recession",
    "CAL": "clinical attachment level",
    "GI": "gingival index",
    "RCT": "root canal treatment",
    "FPD": "fixed partial denture",
    "SRP": "scaling and root planing",
    "OHI": "oral hygiene instruction",
    "FM": "full mouth",
    "BW": "bitewing radiograph",
    "PA": "periapical radiograph",
    "Pano": "panoramic radiograph",
    "OPT": "orthopantomogram",
    "CBCT": "cone-beam computed tomography",
    "MOD": "mesio-occluso-distal",
    "DO": "disto-occlusal",
    "MO": "mesio-occlusal",
    "根管治疗": "root canal treatment",
    "拔除": "extraction",
    "洁治": "scaling",
    "充填": "restoration",
    "烤瓷冠": "porcelain-fused-to-metal crown",
    "活动义齿": "removable denture",
    "固定义齿": "fixed denture",
    "种植": "implant placement",
    "阻生齿": "impacted tooth",
    "龋": "caries",
    "牙髓炎": "pulpitis",
    "根尖周炎": "periapical periodontitis",
    "牙周炎": "periodontitis",
    "冠周炎": "pericoronitis",
    "叩痛": "percussion pain",
    "松动": "mobility",
    "探诊": "probing",
    "冷热刺激痛": "pain on thermal stimulation",
    "低密度影": "radiolucency",
    "阻射影": "radiopacity",
    "牙槽骨吸收": "alveolar bone resorption",
}

_SYSTEM_PROMPT = """\
You are a bilingual dental clinician translating Chinese outpatient dental \
records into English for a public research dataset.

Rules, in priority order:

1. Translate meaning, not words. Produce natural clinical English of the kind a \
dentist would write in a chart note. Do not produce word-by-word glosses.
2. Copy through UNCHANGED: FDI tooth numbers (#36, 36, 11, 16-26), all numeric \
measurements and their units (PD 3-4mm, 0.1g), ICD-10 codes, drug dose strings, \
and bracketed placeholders such as [EMPLOYER], [RELATIVE], [LOCATION]. Never \
renumber a tooth. Never convert units.
3. Expand clinical abbreviations to the English full term given in the glossary.
4. Preserve the clinical hedging of the original. If the Chinese says 考虑 \
("consider"), do not translate it as a definite diagnosis.
5. Do not add information, do not omit information, do not add explanations or \
notes of your own.
6. If a source field is an empty string, return an empty string for it.
7. Keep the segment structure. If the Chinese uses a delimiter such as ；or ，\
to separate findings, use "; " or ", " correspondingly.

Return STRICT JSON only: an object whose keys are exactly the keys you were \
given and whose values are the English strings. No markdown fence, no commentary.
"""


def _glossary_block() -> str:
    return "\n".join(f"{k} = {v}" for k, v in GLOSSARY.items())


@dataclass
class TranslationResult:
    """One record's translations plus the provenance needed to reproduce them."""

    patient_id: str
    translations: dict[str, str]
    model: str
    prompt_sha256: str
    attempts: int = 1
    error: str | None = None


@dataclass
class Translator:
    """Thin, dependency-free client for the DeepSeek chat-completions endpoint."""

    api_key: str
    model: str = DEFAULT_MODEL
    api_url: str = API_URL
    provider: str = "deepseek"
    temperature: float = 0.0
    timeout: float = 180.0
    max_retries: int = 5
    _cache_dir: Path | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, provider: str = "deepseek", **kw) -> "Translator":
        if provider not in PROVIDERS:
            raise ValueError(
                f"unknown provider {provider!r}; expected one of "
                f"{sorted(PROVIDERS)}"
            )
        env_var, url = PROVIDERS[provider]
        key = os.environ.get(env_var) or os.environ.get(env_var.upper())
        if not key:
            raise RuntimeError(
                f"No API key for {provider}: set {env_var} in the environment, "
                "e.g. by sourcing .env"
            )
        return cls(api_key=key, api_url=url, provider=provider, **kw)

    # -- prompt -----------------------------------------------------------
    def build_prompt(self, fields: dict[str, str]) -> str:
        return (
            "Translate every value below into English, following the rules.\n\n"
            + json.dumps(fields, ensure_ascii=False, indent=2)
        )

    def system_prompt(self) -> str:
        """Rules plus glossary.

        The glossary lives here rather than in the user message so that the
        entire prefix is byte-identical across all 463 calls and DeepSeek's
        context cache applies to it; otherwise the table is re-billed at full
        rate once per record.
        """
        return (
            _SYSTEM_PROMPT
            + "\nGlossary (abbreviation = English full term):\n"
            + _glossary_block()
        )

    # -- transport --------------------------------------------------------
    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _complete(self, prompt: str) -> tuple[str, int]:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                body = self._post(payload)
                return body["choices"][0]["message"]["content"], attempt
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                    KeyError, json.JSONDecodeError, OSError) as exc:
                last = exc
                # Auth, malformed request and billing failures will not heal by
                # waiting; retrying them just burns wall-clock on every record.
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (
                    400, 401, 402, 403, 404, 422
                ):
                    raise
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"failed after {self.max_retries} attempts: {last!r}")

    def _complete_with_system(self, system: str, prompt: str) -> tuple[str, int]:
        """Same transport as :meth:`_complete`, with a caller-supplied system
        prompt. Used by the layered pipeline, where the forward and
        back-translation passes need different instructions."""
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "stream": False,
        }
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                body = self._post(payload)
                return body["choices"][0]["message"]["content"], attempt
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                    KeyError, json.JSONDecodeError, OSError) as exc:
                last = exc
                if isinstance(exc, urllib.error.HTTPError) and exc.code in (
                    400, 401, 402, 403, 404, 422
                ):
                    raise
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"failed after {self.max_retries} attempts: {last!r}")

    # -- public API -------------------------------------------------------
    def translate_record(self, record: dict) -> TranslationResult:
        """Translate one clinical record's free-text fields.

        Only non-empty fields are sent. Empty fields map to empty strings
        without spending a token on them.
        """
        pid = str(record.get("patient_id", ""))
        source = {
            f: (record.get(f) or "").strip()
            for f in TRANSLATABLE_FIELDS
        }
        nonempty = {k: v for k, v in source.items() if v}
        out = {f"{k}_en": "" for k in source}

        if not nonempty:
            return TranslationResult(pid, out, self.model, "", attempts=0)

        prompt = self.build_prompt(nonempty)
        sha = hashlib.sha256(
            (self.model + self.system_prompt() + prompt).encode("utf-8")
        ).hexdigest()

        try:
            raw, attempts = self._complete(prompt)
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            return TranslationResult(pid, out, self.model, sha, error=repr(exc))

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            return TranslationResult(
                pid, out, self.model, sha, attempts, error=f"unparseable: {exc}"
            )

        missing = [k for k in nonempty if k not in parsed]
        for k in nonempty:
            val = parsed.get(k, "")
            out[f"{k}_en"] = val.strip() if isinstance(val, str) else ""

        err = f"missing keys: {missing}" if missing else None
        return TranslationResult(pid, out, self.model, sha, attempts, error=err)


# -- verification ---------------------------------------------------------

# Python's \b is Unicode-aware, so it does not fire between an ASCII digit and
# a CJK character -- "#36叩痛" would yield no match. Digit lookarounds instead.
_TOOTH = re.compile(r"(?<![0-9])#?([1-4][1-8])(?![0-9])")


def tooth_numbers(text: str) -> list[str]:
    """FDI tooth numbers appearing in a string, in order.

    A deliberately loose heuristic for translation QA, not a clinical parser:
    any two-digit number in the FDI range counts, so dates and measurements can
    produce false positives. That is harmless here because
    :func:`check_translation` compares the source and translated lists against
    each other -- a false positive present in both cancels out, and only an
    asymmetry between them is reported.
    """
    return _TOOTH.findall(text or "")


def check_translation(source: str, english: str) -> list[str]:
    """Automatic checks on one translated string.

    Catches the failure modes that matter clinically and that a reviewer would
    catch by eye: dropped or invented tooth numbers, an empty translation of
    non-empty source, and untranslated Chinese left in the output.
    """
    problems: list[str] = []
    src, eng = (source or "").strip(), (english or "").strip()

    if src and not eng:
        problems.append("empty translation of non-empty source")
    if not src and eng:
        problems.append("non-empty translation of empty source")

    s_teeth, e_teeth = sorted(tooth_numbers(src)), sorted(tooth_numbers(eng))
    if s_teeth != e_teeth:
        problems.append(f"tooth numbers differ: {s_teeth} vs {e_teeth}")

    residual = [ch for ch in eng if "一" <= ch <= "鿿"]
    if residual:
        problems.append(f"{len(residual)} Chinese characters remain")

    return problems


def audit(records: Iterable[dict]) -> dict:
    """Run :func:`check_translation` across a bilingual corpus."""
    total = flagged = 0
    by_field: dict[str, int] = {}
    examples: list[dict] = []
    for rec in records:
        for f in TRANSLATABLE_FIELDS:
            src, eng = rec.get(f, ""), rec.get(f"{f}_en", "")
            if not (src or eng):
                continue
            total += 1
            probs = check_translation(src, eng)
            if probs:
                flagged += 1
                by_field[f] = by_field.get(f, 0) + 1
                if len(examples) < 40:
                    examples.append(
                        {
                            "patient_id": rec.get("patient_id"),
                            "field": f,
                            "problems": probs,
                            "source": src[:160],
                            "english": eng[:160],
                        }
                    )
    return {
        "strings_checked": total,
        "strings_flagged": flagged,
        "flag_rate": round(flagged / total, 4) if total else 0.0,
        "by_field": by_field,
        "examples": examples,
    }
