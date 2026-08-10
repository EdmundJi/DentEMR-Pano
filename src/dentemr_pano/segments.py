"""Layer 0 of the translation pipeline: protect, segment, deduplicate.

The order of operations matters and is the whole point of this module.

Clinical Chinese carries tokens that must survive translation byte-for-byte:
FDI tooth numbers, ICD-10 codes, drug packaging strings, de-identification
placeholders, measurements. Several of them contain the very characters a naive
segmenter splits on -- ``[5ml:0.1g]`` holds a full stop, ``K04.0`` holds
another. Segmenting first shreds them: an early run of this pipeline produced
fragments like ``0.1g]`` and ``1.7ml]`` as "unique segments" to be translated.

So: **protect, then segment, then deduplicate**. Protected spans are swapped for
opaque sentinels before any splitting happens, travel through translation
untouched, and are restored afterwards.

Deduplication is what makes clinician review affordable. Dental records repeat
themselves heavily -- ``病情变化及时就诊`` ("seek care promptly if the condition
changes") appears in 237 records. Translating and reviewing unique segments
instead of whole records cuts the work by roughly 40%, and the unique-segment
table is the natural unit for the bilingual clinician check in Layer 4.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

#: Patterns whose matches must survive translation unchanged. Order matters:
#: earlier patterns win, so bracketed drug strings are claimed before the
#: bare-number rules can bite into them.
PROTECTED_PATTERNS: tuple[tuple[str, str], ...] = (
    # de-identification placeholders, e.g. [EMPLOYER]
    ("placeholder", r"\[[A-Z_]{3,}\]"),
    # bracketed drug packaging / specimen strings, e.g. [5ml:0.1g], [1.7ml]
    ("drug_spec", r"[\[［][^\[\]［］]{0,40}?[\]］]"),
    # ICD-10 codes, e.g. K04.0, S02.5, K08.x
    ("icd", r"\b[A-TV-Z][0-9]{2}(?:\.[0-9A-Za-z]{1,4})?\b"),
    # FDI tooth notation with an explicit hash, e.g. #36, #11
    ("tooth_hash", r"#\s?[1-4][1-8]"),
    # measurements with units, e.g. PD 3-4mm, 0.5mm, 12s
    ("measurement", r"\b\d+(?:\.\d+)?\s?(?:-\s?\d+(?:\.\d+)?)?\s?(?:mm|cm|ml|mg|g|kV|mA|s)\b"),
    # clinical abbreviations that are already English
    ("abbrev", r"\b(?:RCT|FPD|SRP|OHI|CBCT|BOP|CAL|PD|AL|GR|GI|MOD|DO|MO|BW|PA|OPT|Pano|DR|X)\b"),
)

#: Segment boundaries. Chinese/ASCII sentence punctuation, newlines, and
#: enumerator prefixes such as "1." or "2、" when they start a clause.
_SPLIT = re.compile(r"[；;。\n]+|(?:(?<=^)|(?<=[\s，,；;。]))(?=\d{1,2}\s?[、.)]\s?\D)")

_SENTINEL = "⟦P{}⟧"  # ⟦P0⟧ -- absent from clinical text, survives JSON
_SENTINEL_RE = re.compile(r"⟦P(\d+)⟧")


@dataclass
class Protected:
    """A string with its protected spans replaced by sentinels."""

    masked: str
    tokens: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)

    def restore(self, text: str) -> str:
        """Put the original spans back into a translated string."""
        def swap(m: re.Match) -> str:
            i = int(m.group(1))
            return self.tokens[i] if i < len(self.tokens) else m.group(0)
        return _SENTINEL_RE.sub(swap, text)

    def missing_sentinels(self, text: str) -> list[str]:
        """Sentinels that were in the source but are absent from ``text``."""
        present = {int(m.group(1)) for m in _SENTINEL_RE.finditer(text)}
        return [self.tokens[i] for i in range(len(self.tokens)) if i not in present]


def protect(text: str) -> Protected:
    """Replace every protected span in ``text`` with an opaque sentinel."""
    tokens: list[str] = []
    cats: list[str] = []
    spans: list[tuple[int, int, str, str]] = []

    for cat, pat in PROTECTED_PATTERNS:
        for m in re.finditer(pat, text):
            if any(not (m.end() <= s or m.start() >= e) for s, e, _, _ in spans):
                continue  # overlaps something already claimed
            spans.append((m.start(), m.end(), m.group(0), cat))

    out, last = [], 0
    for start, end, tok, cat in sorted(spans):
        out.append(text[last:start])
        out.append(_SENTINEL.format(len(tokens)))
        tokens.append(tok)
        cats.append(cat)
        last = end
    out.append(text[last:])
    return Protected("".join(out), tokens, cats)


def segment(text: str) -> list[str]:
    """Split masked text into clause-level segments, dropping empties."""
    return [s.strip() for s in _SPLIT.split(text) if s and s.strip()]


def localise(seg: str, parent: Protected) -> tuple[str, list[str], list[str]]:
    """Renumber a segment's sentinels to start at 0, with its own token list.

    Sentinel numbering out of :func:`protect` is scoped to the whole field, so
    the same clause appearing in two records can carry different indices --
    and, worse, the same index can mean different things. Either would make the
    masked string a wrong deduplication key: ``⟦P3⟧`` is not a stable identity.

    Renumbering locally makes the masked segment self-describing. Two clauses
    that differ only in which tooth they name collapse to one translation unit
    (correctly -- the English is identical modulo the token), while each
    occurrence keeps the tokens needed to restore itself.
    """
    idxs = [int(m.group(1)) for m in _SENTINEL_RE.finditer(seg)]
    remap = {old: new for new, old in enumerate(dict.fromkeys(idxs))}
    masked = _SENTINEL_RE.sub(
        lambda m: _SENTINEL.format(remap[int(m.group(1))]), seg
    )
    tokens, cats = [], []
    for old in remap:
        tokens.append(parent.tokens[old] if old < len(parent.tokens) else "")
        cats.append(parent.categories[old] if old < len(parent.categories) else "")
    return masked, tokens, cats


@dataclass
class SegmentTable:
    """The deduplicated segment inventory for a corpus.

    ``unique`` maps a masked segment to the number of times it occurs.
    ``occurrences`` records where each one came from *and the tokens needed to
    restore it*, so translations can be backfilled into the records they belong
    to (Layer 5).
    """

    unique: Counter
    occurrences: list[dict]
    category_counts: Counter

    @property
    def n_unique(self) -> int:
        return len(self.unique)

    @property
    def n_total(self) -> int:
        return sum(self.unique.values())

    @property
    def dedup_rate(self) -> float:
        return 1 - self.n_unique / self.n_total if self.n_total else 0.0


def build_segment_table(records: list[dict], fields: tuple[str, ...]) -> SegmentTable:
    """Protect, segment and deduplicate every translatable field in a corpus."""
    unique: Counter = Counter()
    occurrences: list[dict] = []
    cats: Counter = Counter()

    for rec in records:
        pid = rec.get("patient_id", "")
        for f in fields:
            value = (rec.get(f) or "").strip()
            if not value:
                continue
            p = protect(value)
            cats.update(p.categories)
            for idx, seg in enumerate(segment(p.masked)):
                masked, tokens, tok_cats = localise(seg, p)
                unique[masked] += 1
                occurrences.append(
                    {
                        "patient_id": pid,
                        "field": f,
                        "order": idx,
                        "segment": masked,
                        "tokens": tokens,
                        "token_categories": tok_cats,
                    }
                )
    return SegmentTable(unique, occurrences, cats)


def restore_tokens(text: str, tokens: list[str]) -> str:
    """Substitute an occurrence's own tokens back into a translated segment."""
    def swap(m: re.Match) -> str:
        i = int(m.group(1))
        return tokens[i] if i < len(tokens) else m.group(0)
    return _SENTINEL_RE.sub(swap, text)


def missing_tokens(text: str, tokens: list[str]) -> list[str]:
    """Tokens whose sentinel the translator dropped (Layer 3 token-preservation)."""
    present = {int(m.group(1)) for m in _SENTINEL_RE.finditer(text)}
    return [t for i, t in enumerate(tokens) if i not in present]
