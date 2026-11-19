from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import librosa
import numpy as np

from surgphase.config import UNLABELLED


def _normalize_rows(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=-1, keepdims=True)
    uniform = np.full_like(counts, 1 / counts.shape[-1], dtype=float)
    return np.divide(counts, totals, out=uniform, where=totals > 0)


@dataclass(frozen=True)
class PhaseHMM:
    transition: np.ndarray
    initial: np.ndarray
    prior: np.ndarray

    @classmethod
    def fit(
        cls, label_sequences: Sequence[np.ndarray], num_phases: int, pseudocount: float
    ) -> "PhaseHMM":
        transitions = np.full((num_phases, num_phases), pseudocount, dtype=float)
        initial = np.full(num_phases, pseudocount, dtype=float)
        prior = np.full(num_phases, pseudocount, dtype=float)
        for labels in label_sequences:
            labels = np.asarray(labels, dtype=int)
            labelled = labels != UNLABELLED
            if not labelled.any():
                continue
            consecutive = labelled[:-1] & labelled[1:]
            np.add.at(transitions, (labels[:-1][consecutive], labels[1:][consecutive]), 1)
            initial[labels[labelled][0]] += 1
            prior += np.bincount(labels[labelled], minlength=num_phases)
        return cls(_normalize_rows(transitions), _normalize_rows(initial), _normalize_rows(prior))

    @property
    def num_phases(self) -> int:
        return len(self.prior)

    def _check(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if probabilities.ndim != 2 or probabilities.shape[1] != self.num_phases:
            raise ValueError(
                f"expected probabilities of shape (T, {self.num_phases}), got {probabilities.shape}"
            )
        return probabilities

    def viterbi(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = self._check(probabilities)
        return librosa.sequence.viterbi_discriminative(
            probabilities.T, self.transition, p_state=self.prior, p_init=self.initial
        ).astype(np.int64)

    def filter(self, probabilities: np.ndarray) -> np.ndarray:
        probabilities = self._check(probabilities)
        epsilon = np.finfo(float).tiny
        log_likelihood = np.log(probabilities + epsilon) - np.log(self.prior + epsilon)
        posteriors = np.empty_like(log_likelihood)
        belief = self.initial
        for t, frame in enumerate(log_likelihood):
            log_belief = np.log(belief + epsilon) + frame
            belief = np.exp(log_belief - log_belief.max())
            belief /= belief.sum()
            posteriors[t] = belief
            belief = belief @ self.transition
        return posteriors


def leave_one_out_hmms(
    label_sequences: Mapping[str, np.ndarray], num_phases: int, pseudocount: float
) -> dict[str, PhaseHMM]:
    return {
        held_out: PhaseHMM.fit(
            [labels for key, labels in label_sequences.items() if key != held_out],
            num_phases,
            pseudocount,
        )
        for held_out in label_sequences
    }
