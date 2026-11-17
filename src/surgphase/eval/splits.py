from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    test: str
    val: str
    train: tuple[str, ...]


def leave_one_surgery_out(surgery_ids: Sequence[str]) -> list[Fold]:
    ids = list(surgery_ids)
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate surgery ids in {ids}")
    if len(ids) < 3:
        raise ValueError(f"need at least 3 surgeries for train/val/test folds, got {ids}")
    folds = []
    for position, test in enumerate(ids):
        val = ids[(position + 1) % len(ids)]
        folds.append(Fold(test, val, tuple(s for s in ids if s not in (test, val))))
    return folds
