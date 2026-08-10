#!/usr/bin/env python3
"""Layered zh->en translation of the clinical text fields.

    python scripts/run_translation_pipeline.py path/to/dental_clinical_dataset_v1 \
        --outdir out/translation --batch 25

Layer 0  protect -> segment -> deduplicate
Layer 1  forward translation, two models independently, temperature 0
Layer 2  independent back-translation by the crossed model + embedding similarity
Layer 3  deterministic rule checks (token preservation, negation, length)

Outputs, all under --outdir:

  segments.json          the deduplicated segment table with occurrence counts
  translations.json      per-segment: both models' English, back-translation,
                         similarity, rule report, and whether it needs review
  layer_report.json      the numbers that go into Technical Validation
  review_unique.csv      Layer 4: unique-segment table for bilingual clinician
                         review, ordered by occurrence count so the reviewer's
                         effort lands on the highest-leverage segments first

Every API result is cached per batch under --outdir/.cache, so an interrupted
run resumes without paying twice.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dentemr_pano.pipeline import (  # noqa: E402
    back_system, batch, check_rules, forward_system, normalise_en, parse_batch,
)
from dentemr_pano.segments import build_segment_table  # noqa: E402
from dentemr_pano.translate import GLOSSARY, TRANSLATABLE_FIELDS, Translator  # noqa: E402

FORWARD_PROVIDER, FORWARD_MODEL = "deepseek", "deepseek-v4-flash"
# The second forward pass repeats the same model. Temperature 0 is not
# deterministic on this API, so the pass is not wasted -- but what it measures
# is run-to-run stability, not agreement between two models.
SECOND_PROVIDER, SECOND_MODEL = "deepseek", "deepseek-v4-flash"
# Back-translation deliberately crosses vendors. A round trip through one
# vendor measures that vendor's self-consistency: a model that misreads a term
# on the way out will misread it consistently on the way back and the error
# stays invisible. The back-translator is also given neither the glossary nor
# the Chinese source, so it cannot reconstruct the original from anything but
# the English under test.
BACK_PROVIDER, BACK_MODEL = "qwen", "qwen3.7-plus"


def load_records(root: Path) -> list[dict]:
    files = sorted((root / "clinical_records").glob("*.json"))
    if not files:
        sys.exit(f"no clinical_records/*.json under {root}")
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def run_batches(tr: Translator, system: str, batches, cache: Path,
                tag: str, workers: int) -> dict[int, dict[str, str]]:
    """Translate batches concurrently, caching each by content hash."""
    cache.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict[str, str]] = {}
    todo = []
    for bi, b in enumerate(batches):
        key = hashlib.sha256(
            (tag + tr.provider + tr.model + system
             + json.dumps(b, ensure_ascii=False)).encode()
        ).hexdigest()[:24]
        cp = cache / f"{tag}_{key}.json"
        if cp.exists():
            results[bi] = json.loads(cp.read_text(encoding="utf-8"))
        else:
            todo.append((bi, b, cp))

    print(f"  {tag}: {len(batches)} batches, {len(results)} cached, {len(todo)} to run",
          flush=True)

    def work(item):
        bi, b, cp = item
        payload = {k: v for k, v in b}
        raw, _ = tr._complete_with_system(system, json.dumps(payload, ensure_ascii=False))
        parsed = parse_batch(raw, b)
        cp.write_text(json.dumps(parsed, ensure_ascii=False), encoding="utf-8")
        return bi, parsed

    done = 0
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(work, it): it for it in todo}
            for fut in as_completed(futs):
                bi, it = futs[fut][0], futs[fut]
                try:
                    bi, parsed = fut.result()
                    results[bi] = parsed
                except Exception as exc:  # noqa: BLE001
                    print(f"    batch {bi} failed: {exc!r}", flush=True)
                    results[bi] = {}
                done += 1
                if done % 10 == 0 or done == len(todo):
                    print(f"    {done}/{len(todo)}", flush=True)
    return results


def flatten(batches, results) -> list[str]:
    """Reassemble batch results back into segment order."""
    out = []
    for bi, b in enumerate(batches):
        got = results.get(bi, {})
        for bid, _src in b:
            out.append(got.get(bid, ""))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("out/translation"))
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only N unique segments")
    ap.add_argument("--threshold", type=float, default=0.93)
    ap.add_argument("--embed-model", default="BAAI/bge-small-zh-v1.5")
    ap.add_argument("--skip-second-model", action="store_true",
                    help="run one forward model only (cheaper smoke test)")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    cache = outdir / ".cache"

    # -- Layer 0 ---------------------------------------------------------
    records = load_records(args.dataset)
    st = build_segment_table(records, TRANSLATABLE_FIELDS)
    segs = [s for s, _ in st.unique.most_common()]
    if args.limit:
        segs = segs[: args.limit]
    print(f"Layer 0: {st.n_total:,} segments -> {st.n_unique:,} unique "
          f"({st.dedup_rate:.1%} deduplicated); protected token categories: "
          f"{dict(st.category_counts)}", flush=True)
    (outdir / "segments.json").write_text(json.dumps({
        "n_total": st.n_total, "n_unique": st.n_unique,
        "dedup_rate": st.dedup_rate,
        "protected_token_categories": dict(st.category_counts),
        "unique_segments": [{"segment": s, "count": c} for s, c in st.unique.most_common()],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- Layer 1 ---------------------------------------------------------
    fsys = forward_system(GLOSSARY)
    fbatches = batch(segs, args.batch)
    print(f"Layer 1: forward translation of {len(segs):,} unique segments", flush=True)

    trA = Translator.from_env(provider=FORWARD_PROVIDER, model=FORWARD_MODEL)
    enA = flatten(fbatches, run_batches(trA, fsys, fbatches, cache, "fwdA", args.workers))

    if args.skip_second_model:
        enB = [""] * len(segs)
    else:
        trB = Translator.from_env(provider=SECOND_PROVIDER, model=SECOND_MODEL)
        enB = flatten(fbatches, run_batches(trB, fsys, fbatches, cache, "fwdB", args.workers))

    # -- Layer 2 ---------------------------------------------------------
    print("Layer 2: back-translation + embedding similarity", flush=True)
    bbatches = batch(enA, args.batch)
    trBack = Translator.from_env(provider=BACK_PROVIDER, model=BACK_MODEL)
    zh2 = flatten(bbatches, run_batches(trBack, back_system(), bbatches, cache,
                                        "back", args.workers))

    from sentence_transformers import SentenceTransformer
    import numpy as np
    emb = SentenceTransformer(args.embed_model)
    lens = np.array([len(re.sub(r"\u27e6P\d+\u27e7", "", s)) for s in segs])
    va = emb.encode(segs, normalize_embeddings=True, batch_size=64,
                    show_progress_bar=False)
    vb = emb.encode([z or "" for z in zh2], normalize_embeddings=True,
                    batch_size=64, show_progress_bar=False)
    sim = (va * vb).sum(axis=1)

    # Inter-model agreement. Exact string equality answers "did two models
    # choose identical wording", which is almost never true of free
    # translation and says nothing about whether they agree on meaning
    # ("luxation" vs "dislocation", "Option 1" vs "Plan 1"). Both are
    # reported: the wording figure for transparency, the semantic one as the
    # measure that matters.
    if any(enB):
        ea = emb.encode(enA, normalize_embeddings=True, batch_size=64,
                        show_progress_bar=False)
        eb = emb.encode([e or "" for e in enB], normalize_embeddings=True,
                        batch_size=64, show_progress_bar=False)
        model_sim = (ea * eb).sum(axis=1)
    else:
        model_sim = np.zeros(len(segs))

    # -- Layer 3 ---------------------------------------------------------
    tok_by_seg: dict[str, list[str]] = {}
    for occ in st.occurrences:
        tok_by_seg.setdefault(occ["segment"], occ["tokens"])

    rows, n_rule_pass, n_low_sim, n_model_disagree, n_wording = [], 0, 0, 0, 0
    for i, s in enumerate(segs):
        toks = tok_by_seg.get(s, [])
        rep = check_rules(s, enA[i], toks)
        n_rule_pass += rep.passed
        low = float(sim[i]) < args.threshold
        n_low_sim += low
        wording_differs = bool(enB[i]) and normalise_en(enA[i]) != normalise_en(enB[i])
        disagree = bool(enB[i]) and float(model_sim[i]) < args.threshold
        n_model_disagree += disagree
        n_wording += wording_differs
        rows.append({
            "segment": s, "count": st.unique[s],
            "en_primary": enA[i], "en_secondary": enB[i],
            "back_translation": zh2[i], "similarity": round(float(sim[i]), 4),
            "rules_passed": rep.passed, "rule_detail": rep.detail,
            "dropped_tokens": rep.dropped_tokens,
            "models_disagree": disagree,
            "models_wording_differs": wording_differs,
            "model_similarity": round(float(model_sim[i]), 4),
            "needs_review": bool(low or not rep.passed or disagree),
        })

    (outdir / "translations.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    n = len(segs)
    covered = sum(r["count"] for r in rows)
    report = {
        "layer0": {"segments_total": st.n_total, "segments_unique": st.n_unique,
                   "dedup_rate": round(st.dedup_rate, 4),
                   "protected_token_categories": dict(st.category_counts)},
        "layer1": {"forward_model": FORWARD_MODEL,
                   "secondary_model": None if args.skip_second_model else SECOND_MODEL,
                   "temperature": 0.0, "batch_size": args.batch,
                   "segments_translated": n,
                   "empty_outputs": sum(1 for e in enA if not e.strip())},
        "layer2_note": ("Cosine similarity of sentence embeddings is unstable on "
                        "very short strings, so it is reported stratified by "
                        "source length as well as pooled."),
        "layer2": {"back_translation_model": BACK_MODEL,
                   "back_translation_provider": BACK_PROVIDER,
                   "back_translation_is_blind": True,
                   "back_translation_note": ("Cross-vendor and blind: the "
                       "back-translator receives only the English under test, "
                       "with neither the glossary nor the Chinese source."),
                   "embedding_model": args.embed_model,
                   "threshold": args.threshold,
                   "mean_similarity": round(float(sim.mean()), 4),
                   "median_similarity": round(float(np.median(sim)), 4),
                   "below_threshold": int(n_low_sim),
                   "below_threshold_pct": round(100 * n_low_sim / n, 2),
                   "by_source_length": {
                       band: {
                           "n": int(((lens >= lo) & (lens < hi)).sum()),
                           "mean_similarity": (
                               round(float(sim[(lens >= lo) & (lens < hi)].mean()), 4)
                               if ((lens >= lo) & (lens < hi)).any() else None),
                           "below_threshold": int(
                               (sim[(lens >= lo) & (lens < hi)] < args.threshold).sum()),
                       }
                       for band, lo, hi in (("1-5 chars", 0, 6), ("6-15 chars", 6, 16),
                                            ("16-40 chars", 16, 41), (">40 chars", 41, 10**6))
                   }},
        "layer3": {"rule_pass": int(n_rule_pass),
                   "rule_pass_pct": round(100 * n_rule_pass / n, 2),
                   "token_failures": sum(1 for r in rows if r["dropped_tokens"]),
                   "negation_failures": sum(1 for r in rows if "negation" in r["rule_detail"]),
                   "length_failures": sum(1 for r in rows if "len ratio" in r["rule_detail"]),
                   "bare_tooth_failures": sum(1 for r in rows if "bare teeth" in r["rule_detail"])},
        "repeat_run_consistency": {
            "pass_1_model": FORWARD_MODEL,
            "pass_2_model": SECOND_MODEL,
            "same_model": FORWARD_MODEL == SECOND_MODEL,
            "semantic_divergence_count": int(n_model_disagree),
            "semantic_divergence_pct": round(100 * n_model_disagree / n, 2),
            "mean_similarity": round(float(model_sim.mean()), 4),
            "wording_differs_count": int(n_wording),
            "wording_differs_pct": round(100 * n_wording / n, 2),
            "note": ("Two independent forward passes. When both use the same "
                     "model this measures run-to-run stability, not agreement "
                     "between models: temperature 0 is not deterministic on "
                     "this API. Wording divergence counts any phrasing "
                     "difference and is high by construction for free "
                     "translation; the semantic figure is the one to "
                     "interpret.")},
        "review_queue": {"unique_segments": sum(1 for r in rows if r["needs_review"]),
                         "occurrences_covered": covered},
    }
    (outdir / "layer_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- Layer 4 input ---------------------------------------------------
    with (outdir / "review_unique.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "occurrences", "chinese", "english", "back_translation",
                    "similarity", "auto_flag", "verdict_0_1_2", "correction", "error_type"])
        for i, r in enumerate(sorted(rows, key=lambda x: -x["count"]), 1):
            flag = []
            if r["similarity"] < args.threshold:
                flag.append("low_sim")
            if not r["rules_passed"]:
                flag.append(r["rule_detail"])
            if r["models_disagree"]:
                flag.append("model_disagree")
            w.writerow([i, r["count"], r["segment"], r["en_primary"],
                        r["back_translation"], r["similarity"], "; ".join(flag), "", "", ""])

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
