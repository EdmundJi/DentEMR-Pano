"""The 23-item, 5-dimension quality checklist used for technical validation.

Encoding the rubric here lets the reliability analysis pick the correct
agreement coefficient per item automatically: ordinal items get a weighted
coefficient, binary items get an unweighted one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    """One checklist item.

    Attributes:
        code: Item identifier, ``"<dim>.<n>"``.
        name: Short item name as printed in the rubric table.
        scale: Admissible scores. Ordinal items allow partial credit (0/1/2);
            binary items allow only 0 or 2.
    """

    code: str
    name: str
    scale: tuple[int, ...]

    @property
    def dimension(self) -> str:
        return f"Dim{self.code.split('.')[0]}"

    @property
    def binary(self) -> bool:
        return self.scale == (0, 2)

    @property
    def max_score(self) -> int:
        return max(self.scale)


ORDINAL = (0, 1, 2)
BINARY = (0, 2)

DIMENSION_NAMES: dict[str, str] = {
    "Dim1": "Image Diagnostic Quality",
    "Dim2": "Text Completeness",
    "Dim3": "Clinical Consistency",
    "Dim4": "Coding Accuracy",
    "Dim5": "De-identification Integrity",
}

ITEMS: tuple[Item, ...] = (
    # Dimension 1 -- Image Diagnostic Quality (max 10)
    Item("1.1", "Anatomical Coverage", ORDINAL),
    Item("1.2", "Contrast & Density", ORDINAL),
    Item("1.3", "Positioning", ORDINAL),
    Item("1.4", "Artifact Control", ORDINAL),
    Item("1.5", "Image-Record Match", BINARY),
    # Dimension 2 -- Text Completeness (max 14)
    Item("2.1", "Chief Complaint", ORDINAL),
    Item("2.2", "Present Illness", ORDINAL),
    Item("2.3", "Oral Exam", ORDINAL),
    Item("2.4", "Imaging", ORDINAL),
    Item("2.5", "Diagnosis", ORDINAL),
    Item("2.6", "Treatment Plan", ORDINAL),
    Item("2.7", "Procedure/Advice", ORDINAL),
    # Dimension 3 -- Clinical Consistency (max 8)
    Item("3.1", "CC to Dx", ORDINAL),
    Item("3.2", "Exam to Dx", ORDINAL),
    Item("3.3", "Dx to Tx", ORDINAL),
    Item("3.4", "Multi-Dx Coherence", ORDINAL),
    # Dimension 4 -- Coding Accuracy (max 6)
    Item("4.1", "ICD-10 Presence", ORDINAL),
    Item("4.2", "Code-Dx Match", ORDINAL),
    Item("4.3", "Code Format", ORDINAL),
    # Dimension 5 -- De-identification Integrity (max 8, binary)
    Item("5.1", "No Direct ID", BINARY),
    Item("5.2", "No Insurance/MRN", BINARY),
    Item("5.3", "No Kin/Employer", BINARY),
    Item("5.4", "No Exact Dates", BINARY),
)

BY_CODE: dict[str, Item] = {i.code: i for i in ITEMS}

#: Items grouped by dimension, in rubric order.
BY_DIMENSION: dict[str, tuple[Item, ...]] = {
    dim: tuple(i for i in ITEMS if i.dimension == dim) for dim in DIMENSION_NAMES
}

#: Maximum attainable score per dimension and overall.
DIMENSION_MAX: dict[str, int] = {
    dim: sum(i.max_score for i in items) for dim, items in BY_DIMENSION.items()
}
COMPOSITE_MAX: int = sum(DIMENSION_MAX.values())

assert len(ITEMS) == 23, "checklist must contain 23 items"
assert COMPOSITE_MAX == 46, f"composite maximum must be 46, got {COMPOSITE_MAX}"
