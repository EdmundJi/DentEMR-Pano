# What this repository does not contain, and why

The editor's request was to publish "a copy of the full code … excluding the
elements that relate to the EMR". This document draws that line explicitly, so
that a reader can tell the difference between code that is absent because it
was never written and code that is absent because it is sensitive.

## The one private input: the identifier map

Release v3 replaced every `patient_id` (and image file name) by a random
study identifier, because the identifiers used during collection had been
assigned at the hospital workstation and could not be shown to be unrelated
to hospital numbering. `scripts/build_v3.py` reads the collection-to-release
identifier map from `--id-map`; that file is held by the principal
investigator and is not published, so a rebuild without it yields a release
with different (equally valid) random identifiers.

## Nothing else is withheld

An earlier version of this file said that a "de-identification pattern
library" was withheld. That was inaccurate and has been removed. No such
library exists: identifiers were kept out of the dataset at transcription
time (the field template has no slot for names, identity numbers, contact
details or addresses, and the transcribers did not copy them from the
narrative); dates written by physicians into the narratives were reduced to
year-month granularity during release preparation; and the result was
checked by manual review and by the generic residual-identifier probes in
`validate.py`. Those probes are the complete verification code and are
published here. `residual_identifier_scan.json` inside each release copy is
the output of running them over the whole corpus.

## Outside the repository boundary

The dataset was assembled by hand, so there is no extraction pipeline to
publish:

| Stage | Status |
|---|---|
| Retrieval of the record page (HIS workstation) and of the radiograph (PACS viewer window capture) | manual, per the collection guide summarised in the paper's Methods; no code |
| Transcription (double entry by trained undergraduates) and reconciliation (two dental-student authors) | manual; no code, and the per-cell correction history was not logged |
| Assignment of `chief_complaint_category`, `primary_diagnosis_icd`, `treatment_category` | manual, by the two reconciling authors under a shared written rule; no code |
| Schema definition | `schema.py` |
| Dataset validation, residual-identifier scan, duplicate-image check | `validate.py` |
| Radiograph label masking, grayscale conversion, release assembly | `scripts/build_v2.py`, `scripts/split_language_copies.py` |
| v3 linkage repair and archive assembly | `scripts/build_v3.py` |
| English translation pipeline and checks | `translate.py`, `segments.py`, `pipeline.py`, `scripts/run_translation_pipeline.py` |
| Cohort tables for the descriptor | `scripts/make_descriptor_tables.py` |
| Inter-rater reliability with confidence intervals | `reliability.py`, `ratings.py`, `scripts/run_reliability.py` |
| Figures 1 and 2 | `scripts/make_workflow_figure.py`, `scripts/make_hero_figure.py` |

The modules here are therefore a **reference implementation** of the
schema, the rubric and the statistics. They reproduce every number quoted in
the Data Records and Technical Validation sections from the released archive
(plus the rater score table for the reliability analysis). They are not, and
do not claim to be, a recovered copy of a hospital pipeline, because the
dataset was not built by one.

## What a reader can actually reproduce

Given the released archives from Figshare (the rater score table ships inside
each archive as `quality_scores.csv`):

- every structural, completeness, de-identification and duplicate-image check
  reported about the dataset;
- the per-field completeness table, the ICD-10 code table and the
  per-physician documentation measures;
- every inter-rater reliability coefficient with its confidence interval,
  including the zero-filled values of the original submission (re-introduce
  the zero fill to reproduce them);
- the deterministic translation consistency checks described in Methods;
- the v3 archives from the v2 archives.

That covers all quantitative claims in the descriptor. It does not cover the
construction of the archive from the hospital systems, which was manual and is
described in the Methods rather than shipped as code.
