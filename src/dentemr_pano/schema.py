"""Canonical schema for the DentEMR-Pano clinical records.

This module is the single source of truth for the released record structure.
The manuscript's schema table is generated from :data:`FIELDS` (see
``scripts/make_schema_table.py``) so the paper and the data cannot drift apart
again -- the v1 release and the submitted Table 4 disagreed on five field names
and on two fields that were described but never shipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Requirement = Literal["required", "expected", "optional", "derived"]


@dataclass(frozen=True)
class Field:
    """One field of a clinical record.

    Attributes:
        name: Key as it appears in the released JSON.
        source: Originating EMR section header, or a non-EMR provenance note.
        kind: Coarse value type used by the validator.
        requirement: ``required`` fields must be non-empty in every record;
            ``expected`` fields are mandatory per the documentation template but
            are empty in a minority of authentic records; ``optional`` may be
            empty; ``derived`` is not copied from the EMR at all.
        language: ``zh`` for free-text Chinese narrative, ``en`` for English,
            ``code`` for controlled vocabulary, ``none`` for numerics/ids.
        description: Human-readable description used in generated docs.
    """

    name: str
    source: str
    kind: Literal["string", "integer", "enum", "code", "filename"]
    requirement: Requirement
    language: Literal["zh", "en", "code", "none"]
    description: str

    @property
    def translatable(self) -> bool:
        """Whether the field carries Chinese free text needing an ``_en`` twin."""
        return self.language == "zh" and self.kind == "string"


#: Fields in released order. Names match ``clinical_records/*.json`` exactly.
FIELDS: tuple[Field, ...] = (
    Field("patient_id", "PACS/HIS linkage", "string", "required", "none",
          "De-identified case identifier; primary key."),
    Field("image_file", "PACS linkage", "filename", "optional", "none",
          "Paired panoramic radiograph filename; empty when no image exists."),
    Field("has_image", "derived", "enum", "required", "none",
          "'yes' when a paired radiograph is released, otherwise 'no'."),
    Field("age", "HIS demographics", "integer", "required", "none",
          "Patient age in years at the visit (18-90)."),
    Field("sex", "HIS demographics", "enum", "required", "code",
          "Patient sex, recorded as 男 (male) / 女 (female)."),
    Field("attending_physician", "HIS provider field", "enum", "required", "none",
          "Anonymised contributing physician code, P1-P5."),
    Field("chief_complaint", "【主诉】", "string", "expected", "zh",
          "Presenting symptom and duration in the patient's own words."),
    Field("chief_complaint_en", "derived", "enum", "expected", "en",
          "Coarse English category of the chief complaint; not a translation."),
    Field("history_of_present_illness", "【现病史】", "string", "expected", "zh",
          "Onset, triggers and progression narrative."),
    Field("oral_examination", "【口内检查】", "string", "expected", "zh",
          "Intraoral findings using FDI tooth notation."),
    Field("imaging_examination", "【辅助检查】", "string", "expected", "zh",
          "Panoramic radiograph interpretation."),
    Field("primary_diagnosis", "【诊断】", "string", "required", "zh",
          "Primary clinical diagnosis text."),
    Field("primary_diagnosis_icd", "【诊断】", "code", "required", "code",
          "Semicolon-separated ICD-10 codes for the recorded diagnoses."),
    Field("treatment_category", "derived", "enum", "optional", "en",
          "Standardised English treatment category, semicolon-separated."),
    Field("treatment_plan", "【治疗计划/处理】", "string", "expected", "zh",
          "Planned treatment strategy and consented option."),
    Field("procedure", "HIS order list", "string", "expected", "zh",
          "Billing-linked order list of examinations, procedures and drugs."),
    Field("physician_advice", "【医嘱】", "string", "expected", "zh",
          "Follow-up instructions and patient education."),
)

BY_NAME: dict[str, Field] = {f.name: f for f in FIELDS}

#: Chinese free-text fields that receive an ``_en`` counterpart in v2.
TRANSLATABLE: tuple[str, ...] = tuple(f.name for f in FIELDS if f.translatable)

#: Fields present in the v1 release that are dropped from the public v2 archive.
#:
#: ``notes``           -- empty string in all 463 records, carries no information.
#: ``routine_checkup`` -- documented as a routine-check-up flag but holds 463
#:                        distinct integers, i.e. a residual per-record sequence
#:                        number. It is a quasi-identifier with no clinical
#:                        meaning and is removed rather than re-documented.
DROPPED_IN_V2: dict[str, str] = {
    "notes": "empty in all 463 records",
    "routine_checkup": "residual per-record sequence number, not a check-up flag",
}

#: Physician case counts in the v1 release, used to check stratification.
PHYSICIANS: tuple[str, ...] = ("P1", "P2", "P3", "P4", "P5")

#: Fields the submitted manuscript's Table 4 described but which the release
#: does not contain. Recorded here so the revision can resolve each explicitly.
DESCRIBED_BUT_ABSENT: dict[str, str] = {
    "past_medical_history": "Table 4 lists 【既往史/过敏史】 as an optional field; "
                            "no such key exists in any released record.",
    "secondary_diagnoses": "Table 4 and a dedicated Methods paragraph describe "
                           "expert-reviewed secondary diagnoses; the release "
                           "encodes additional diagnoses only as extra ICD "
                           "codes inside primary_diagnosis_icd.",
}

#: Manuscript field names that were renamed in the release, old -> new.
RENAMED_SINCE_SUBMISSION: dict[str, str] = {
    "present_illness": "history_of_present_illness",
    "imaging_description": "imaging_examination",
    "recommendations": "physician_advice",
    "radiograph_filename": "image_file",
    "procedures_performed": "procedure",
}
