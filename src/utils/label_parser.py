from __future__ import annotations

import re
from typing import Iterable, Sequence

import numpy as np  # type: ignore[import]

CLASS_CODES = ("D", "G", "C", "A", "H", "M", "O")

# Key phrases mapped to clinical classes; substrings are matched case-insensitively.
PHRASE_MAP: dict[str, Sequence[str]] = {
    "D": (
        r"diabetic",
        r"retinopathy",
        r"npdr",
        r"pdr",
        r"macular edema",
        r"neovascular",
    ),
    "G": (
        r"glaucoma",
        r"optic atrophy",
        r"cup to disc",
    ),
    "C": (
        r"cataract",
        r"lens opacity",
    ),
    "A": (
        r"age[- ]?related",
        r"amd",
        r"macular degeneration",
    ),
    "H": (
        r"hypertensive",
        r"arterio",
        r"arteriosclerosis",
    ),
    "M": (
        r"pathological myopia",
        r"high myopia",
        r"myopic",
    ),
    "O": (
        r"drusen",
        r"epiretinal membrane",
        r"vitreous",
        r"laser spot",
        r"macular hole",
        r"central serous",
        r"sheen",
        r"atrophy",
        r"hemorrhage",
        r"uveitis",
        r"myelinated",
        r"retinitis",
    ),
}

NEGATIVE_TERMS = (
    r"normal fundus",
    r"no obvious",
    r"no significant",
    r"normal",
)

PUNCT_TRANSLATIONS = str.maketrans({"，": ",", "；": ",", ";": ",", "|": ","})


def _normalised_terms(text: str) -> str:
    cleaned = text.lower().translate(PUNCT_TRANSLATIONS)
    cleaned = re.sub(r"[^a-z0-9, ]+", " ", cleaned)
    cleaned = re.sub(r", +", ",", cleaned)
    return cleaned


def _detect_positive_classes(text: str) -> set[str]:
    pos: set[str] = set()
    if not text:
        return pos
    normalised = _normalised_terms(text)
    if any(term in normalised for term in NEGATIVE_TERMS):
        return pos
    for cls, phrases in PHRASE_MAP.items():
        for phrase in phrases:
            if re.search(phrase, normalised):
                pos.add(cls)
                break
    if not pos and normalised.strip():
        pos.add("O")
    return pos


def keywords_to_multihot(text: str | float | None, classes: Iterable[str] = CLASS_CODES) -> np.ndarray:
    """Convert diagnostic keywords into a multi-hot vector for the specified classes."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        positives: set[str] = set()
    else:
        positives = _detect_positive_classes(str(text))
    ordered_classes = tuple(classes)
    vector = np.zeros(len(ordered_classes), dtype=np.float32)
    for idx, cls in enumerate(ordered_classes):
        vector[idx] = 1.0 if cls in positives else 0.0
    return vector
