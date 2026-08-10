# DentEMR-Pano

Validation and analysis code for **DentEMR-Pano**, a dental clinical dataset
pairing structured EMR narratives with panoramic radiographs: 463 outpatient
cases from five physicians at the First Affiliated Hospital of Shihezi
University, 457 of them with a de-identified panoramic radiograph.

- Dataset: <https://doi.org/10.6084/m9.figshare.32219928> (CC BY 4.0)
- Data descriptor: submitted to *Scientific Data*

> **This is a reference implementation, not the original pipeline.** Records
> were exported from the hospital's HIS system and structured manually by two
> trained research assistants following the extraction protocol in the paper's
> Methods, with discrepancies reconciled by a supervising clinician;
> radiographs were exported from PACS and de-identified. Hospital-side export
> and retrieval components belong to the clinical source systems and are not
> published. The code here encodes the same schema, rubric and statistics and
> reproduces the published numbers. See
> [`docs/EXCLUSIONS.md`](docs/EXCLUSIONS.md) for the stage-by-stage breakdown
> and for the one thing withheld deliberately (the de-identification pattern
> library).

## Install

```bash
git clone https://github.com/EdmundJi/DentEMR-Pano.git
cd DentEMR-Pano
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Download and unpack the dataset archives from Figshare. The v2 release ships
as two synchronized language copies; the Chinese copy is the source record of
reference and is the default input for the tools below:

```
dental_clinical_dataset_v2_zh/      # Chinese source copy
├── clinical_records/       463 × {patient_id}.json
└── panoramic_radiographs/  457 × {patient_id}_panorama.png
dental_clinical_dataset_v2_en/      # English copy (narrative fields suffixed _en)
```

## Validate the archive

```bash
dentemr-validate path/to/dental_clinical_dataset_v2_zh --json audit.json
```

Checks the schema against `schema.py`, image–record pairing in both directions,
per-field completeness, ICD-10 code well-formedness, demographic bounds against
the stated inclusion criteria, and residual identifiers in every free-text
field. Exits non-zero on any error.

## Reproduce the cohort statistics and figures

```bash
python scripts/run_cohort.py path/to/dental_clinical_dataset_v2_zh --outdir out
```

Writes the three dataset-characteristics figures (PDF + PNG at 600 dpi), the
counts behind each as CSV, the chi-square tests of disease prevalence against
sex with Holm-Bonferroni correction, and a `cohort.json` holding every cohort
number quoted in the paper.

Age bands are half-open, `[low, high)`. Disease categories are multi-label: 127
of 463 cases carry more than one ICD code, so category counts sum to more than
the cohort size and percentages are reported over cases.

## Reproduce the inter-rater reliability analysis

```bash
python scripts/run_reliability.py ratings.csv --outdir out
```

`ratings.csv` holds the three raters' item-level scores in either a long layout
(`case_id,rater,item,score`) or a wide one (`case_id,rater,1.1,…,5.4`). Generate
a blank template with the right shape:

```bash
python scripts/make_ratings_template.py --dataset path/to/dental_clinical_dataset_v2_zh
```

Outputs per-item and per-dimension coefficients, and a LaTeX fragment that the
manuscript includes directly so the printed table cannot drift from the
computed numbers.

### Which coefficient, and why more than one

| Statistic | Reported at | Purpose |
|---|---|---|
| ICC(2,1), ICC(2,k) | dimension totals, composite | two-way random effects, absolute agreement — the reliability of a single rater and of the three-rater mean |
| Gwet's AC1 | binary items, Dim5 | chance-corrected agreement that stays defined when the marginal distribution is concentrated |
| Gwet's AC2 | ordinal items, pooled dimensions | AC1 with quadratic weights, so scoring 1 against 2 counts as a smaller disagreement than 0 against 2 |
| PABAK | all | a simpler prevalence-independent reference point |
| Exact agreement | all | the raw unanimity rate, uncorrected |

ICC is a variance ratio. Where nearly every case receives the same score,
between-case variance approaches zero and the ratio collapses even though the
raters agree almost everywhere; where it is exactly zero the statistic is
undefined rather than perfect. Three of the five checklist dimensions sit in
that regime and one is degenerate outright, so a chance-corrected coefficient is
reported alongside throughout.

The chance-corrected coefficients for a dimension are computed on its **pooled
item scores**, not on the dimension total. Applying quadratic weights to a
47-point composite would treat almost every pair of scores as near-agreement and
drive the coefficient toward 1 regardless of the data.

## Tests

```bash
pytest
```

ICC reproduces the published values for the Shrout & Fleiss (1979) worked
example (ICC(2,1) = 0.29, ICC(2,k) = 0.62). AC1 is checked against a
hand-computed four-subject table and against the all-maximum case that Dim5
produces, where ICC is undefined and AC1 is exactly 1.00. The suite also pins
the identity `k·I₁/(1+(k−1)·I₁) = I_k`, which holds exactly for this model.

## Layout

```
src/dentemr_pano/
├── schema.py       field definitions; single source of truth for the
│                   validator and the manuscript's schema table
├── validate.py     structural, completeness and de-identification checks
├── checklist.py    the 23-item, 5-dimension quality rubric
├── reliability.py  ICC, Gwet's AC1/AC2, PABAK
├── ratings.py      rater score loading and reshaping
├── cohort.py       demographics, disease spectrum, hypothesis tests
├── figures.py      the dataset-characteristics figures
└── translate.py    zh->en translation of the clinical text fields
scripts/            command-line drivers
tests/              correctness tests against published reference values
docs/EXCLUSIONS.md  what is withheld and why
```

## Citation

If you use the dataset or this code, please cite the data descriptor and the
dataset itself:

> Ji, Y., Du, Y., Wang, S., Song, J., Chen, Y., Chen, Q. & Zhang, R.
> A multi-physician dental clinical dataset with structured EMR narratives and
> paired panoramic radiographs. *Scientific Data* (submitted).

> Ji, Y. *et al.* DentEMR-Pano. *figshare* <https://doi.org/10.6084/m9.figshare.32219928> (2026).

## License

Code: MIT (see [`LICENSE`](LICENSE)). Dataset: CC BY 4.0, distributed separately
via Figshare. Patient re-identification is prohibited.
