# DentEMR-Pano

Validation and analysis code for **DentEMR-Pano**, a dental clinical dataset
pairing structured EMR narratives with panoramic radiographs: 463 outpatient
cases from five physicians at the First Affiliated Hospital of Shihezi
University, 443 of them with a de-identified panoramic radiograph (release v3).

- Dataset: <https://doi.org/10.6084/m9.figshare.32219928.v3> (CC BY 4.0)
- Data descriptor: submitted to *Scientific Data*

> **This is a reference implementation, not a recovered pipeline.** The
> dataset was assembled by hand. Records were photographed from the hospital's
> HIS workstation screen and typed into a field template independently by two
> trained undergraduate transcribers per record (no OCR); two dental-student
> authors then reconciled the two copies against the photographed record and
> assigned the category and ICD-10 fields. Radiographs were saved from the
> PACS viewer with its window-capture function as 8-bit bitmaps; no DICOM
> object or header was exported. There is therefore no extraction pipeline to
> publish. The code here encodes the same schema, rubric and statistics and
> reproduces the published numbers from the released archive. See
> [`docs/EXCLUSIONS.md`](docs/EXCLUSIONS.md) for the stage-by-stage
> breakdown. Nothing is withheld except the private map between collection
> identifiers and the random study identifiers of v3.

## Install

```bash
git clone https://github.com/EdmundJi/DentEMR-Pano.git
cd DentEMR-Pano
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Download and unpack the dataset archives from Figshare. The v3 release ships
as two synchronized language copies; the Chinese copy is the source record of
reference:

```
dental_clinical_dataset_v3_zh/      # Chinese source copy
├── clinical_records/       463 × {patient_id}.json
├── panoramic_radiographs/  443 × {patient_id}_panorama.png
├── metadata.csv · README.md · CHANGELOG.md · build_report.json
├── quality_scores.csv      160 cases × 3 raters × 23 items (long layout)
└── residual_identifier_scan.json
dental_clinical_dataset_v3_en/      # English copy (narrative fields suffixed _en)
└── … plus translation_screening_log.json
```

## Validate the archive

```bash
dentemr-validate path/to/dental_clinical_dataset_v3_zh --json audit.json
dentemr-validate path/to/dental_clinical_dataset_v3_en --json audit_en.json
```

The validator detects which language copy it is given. It checks the schema
against `schema.py`, image–record pairing in both directions, **byte-identical
image content shared by more than one record** (the check that found the seven
cross-record duplicates repaired in v3), per-field completeness, ICD-10 code
well-formedness, demographic bounds against the stated inclusion criteria, and
residual identifiers in every free-text field (exact dates, national identity
numbers, mobile numbers, unreplaced placeholders). It exits non-zero on any
error. `residual_identifier_scan.json` in each release copy is its output.

## Reproduce the Data Records tables

```bash
python scripts/make_descriptor_tables.py path/to/dental_clinical_dataset_v3_zh --outdir tables
```

Writes the per-field completeness table, the ICD-10 code table (with the
precision at which each code is released) and the per-physician documentation
measures (narrative length, moving-average type-token ratio, shorthand density)
as LaTeX fragments plus a JSON with every number.

## Reproduce the inter-rater reliability analysis

```bash
python scripts/run_reliability.py path/to/dental_clinical_dataset_v3_zh/quality_scores.csv --outdir out
```

`quality_scores.csv` ships inside each release archive and holds the three
raters' item-level scores for the 160 evaluated cases (`case_id` E001–E160,
`patient_id` for the 158 cases in the release, blank for the two excluded
cases). A rating file is accepted in either a long layout
(`case_id,rater,item,score`) or a wide one (`case_id,rater,1.1,…,5.4`). Generate
a blank template with the right shape:

```bash
python scripts/make_ratings_template.py --dataset path/to/dental_clinical_dataset_v3_zh
```

Outputs per-item and per-dimension coefficients with 95% confidence intervals,
the manuscript's reliability table as a LaTeX fragment, the per-item table
(Supplementary Table S3) as LaTeX and xlsx, and a JSON of everything.

### Which coefficient, and why more than one

| Statistic | Reported at | Interval | Purpose |
|---|---|---|---|
| ICC(2,1), ICC(2,k) | dimension totals, composite | exact F-based (McGraw & Wong 1996); ICC(2,k) by Spearman–Brown transform | two-way random effects, absolute agreement |
| Gwet's AC1 | binary items, Dim5 | Gwet's linearised variance | chance-corrected agreement that stays defined when the marginal distribution is concentrated |
| Gwet's AC2 | ordinal items, pooled dimensions | Gwet's linearised variance | AC1 with quadratic weights |
| PABAK | all | normal approximation from the per-subject agreement | prevalence-independent reference point |
| Exact agreement | all | — | raw unanimity rate |

ICC is a variance ratio. Where nearly every case receives the same score,
between-case variance approaches zero and the ratio collapses even though the
raters agree almost everywhere; where it is exactly zero the statistic is
undefined rather than perfect.

The chance-corrected coefficients for a dimension are computed on its **pooled
item scores**, not on the dimension total, and the per-item coefficients are
reported alongside so the pooled value can be checked against its parts.
Applying quadratic weights to the 46-point composite would treat almost every
pair of scores as near-agreement and drive the coefficient toward 1 regardless
of the data.

## Rebuild v3 from v2

```bash
python scripts/build_v3.py path/to/v2_release_dir --outdir v3 \
    --id-map private/id_map_v2_to_v3.csv --quality-scores quality_scores_v2_ids.csv
```

Replaces every `patient_id` and image file name by a random study identifier
(`DP` + six digits; the map is created with fresh random numbers if the file
does not exist and is never shipped), unlinks the 14 records whose image file
was byte-identical to another record's, reduces two month-day mentions to
month granularity, re-keys and ships `quality_scores.csv`, regenerates the
metadata and documentation, runs the validator on both copies, verifies
record by record that nothing else differs from v2, and writes the two release
archives. The identifier map is private: without it a rebuild produces a
release with different identifiers.

## Tests

```bash
pytest
```

ICC reproduces the published values for the Shrout & Fleiss (1979) worked
example (ICC(2,1) = 0.29, ICC(2,k) = 0.62), and its confidence interval
reproduces the reference values for that example (0.019–0.761 and
0.071–0.927). AC1 is checked against a hand-computed four-subject table and
against the all-maximum case that Dim5 produces, where ICC is undefined and
AC1 is exactly 1.00.

## Layout

```
src/dentemr_pano/
├── schema.py       field definitions; single source of truth for the
│                   validator and the manuscript's schema table
├── validate.py     structural, completeness, de-identification and
│                   duplicate-image checks (language-aware)
├── checklist.py    the 23-item, 5-dimension quality rubric
├── reliability.py  ICC with F-based CI, Gwet's AC1/AC2, PABAK
├── ratings.py      rater score loading and reshaping
├── cohort.py       demographics and disease-spectrum counts
├── figures.py      cohort figures (no longer in the descriptor)
├── segments.py     segmentation and protected-token masking
├── pipeline.py     translation pipeline checks (tooth numbers, negation)
└── translate.py    zh->en translation of the clinical text fields
scripts/            command-line drivers (see above)
tests/              correctness tests against published reference values
docs/EXCLUSIONS.md  what is outside the repository boundary and why
```

## Citation

If you use the dataset or this code, please cite the data descriptor and the
dataset itself:

> Ji, Y., Du, Y., Wang, S., Song, J., Chen, Y., Chen, Q. & Zhang, R.
> A multi-physician dental clinical dataset with structured EMR narratives and
> paired panoramic radiographs. *Scientific Data* (submitted).

> Ji, Y. *et al.* DentEMR-Pano (version 3). *figshare* <https://doi.org/10.6084/m9.figshare.32219928.v3> (2026).

## License

Code: MIT (see [`LICENSE`](LICENSE)). Dataset: CC BY 4.0, distributed separately
via Figshare. Patient re-identification is prohibited.
