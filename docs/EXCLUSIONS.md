# What this repository does not contain, and why

The editor's request was to publish "a copy of the full code … excluding the
elements that relate to the EMR". This document draws that line explicitly, so
that a reader can tell the difference between code that is absent because it is
sensitive and code that is absent because it was never part of the public
release.

## Withheld deliberately

### 1. De-identification pattern library

The regular expressions and replacement rules used to detect and remove direct
identifiers (names, national identity numbers, insurance and medical record
numbers, telephone numbers) and indirect identifiers (employer names, kinship
terms, residential districts) are withheld.

A de-identification rule set is a map of what the scrubber looks for. Anyone
holding both the rule set and a de-identified corpus can enumerate what the
scrubber would *not* have caught, which is a re-identification aid rather than a
reproducibility aid.

What is published instead is the **verification** side: `validate.py` ships a
generic residual-identifier scanner that any user can run against the released
archive to confirm the de-identification held. Finding the leaks is the part
that belongs in public; the institutional pattern library is not.

## Outside the repository boundary

The repository does not contain hospital-side export or retrieval code (HIS
record export, PACS/DICOM retrieval), because those components belong to the
clinical source systems and are not part of the public release. Structured
field extraction was performed manually by trained research assistants
following the protocol described in the paper's Methods; being manual, there
is no extraction pipeline to publish.

The modules here are therefore a **reference implementation**: they encode the
same schema, the same rubric and the same statistics as the published dataset,
and they reproduce its numbers, but they are not a recovered copy of the
hospital workflow, because that workflow is outside the repository boundary.

Specifically:

| Stage | Status |
|---|---|
| Hospital-side export and retrieval (HIS, PACS/DICOM) | not included; belongs to the clinical source systems |
| Field segmentation and structured extraction | manual, per the Methods protocol; no pipeline existed to publish |
| Schema definition | reimplemented (`schema.py`) |
| Dataset validation and QC | newly written (`validate.py`) |
| Image de-identification masking and release packaging | encoded in the release-assembly scripts (`scripts/build_v2.py`, `scripts/split_language_copies.py`) |
| English translation pipeline and checks | included (`translate.py`, `segments.py`, `scripts/run_translation_pipeline.py`) |
| Cohort statistics and figures | reimplemented (`cohort.py`, `figures.py`) |
| Inter-rater reliability | reimplemented (`reliability.py`, `ratings.py`) |

This distinction is stated plainly in the README and in the manuscript's Code
Availability section. Presenting the repository as the hospital workflow
itself would be a misrepresentation, and the difference is checkable by anyone
who compares the public code against the release package.

## What a reader can actually reproduce

Given the released archives from Figshare and, for the reliability analysis,
the rater score table:

- every structural, completeness and de-identification check reported about the
  dataset;
- every cohort statistic and all three characteristics figures;
- every inter-rater reliability coefficient, including the ICC values in the
  original submission and the Gwet's AC1/AC2 and PABAK values added in revision;
- the deterministic translation consistency checks described in Methods.

That covers all quantitative claims in the Data Records and Technical Validation
sections. It does not cover the construction of the archive, which was manual
and required access to the hospital's systems — and which, being manual, is
described in the Methods rather than shipped as code.
