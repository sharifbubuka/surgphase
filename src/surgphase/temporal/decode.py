from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from surgphase.config import TEMPORAL_METHODS
from surgphase.temporal.hmm import PhaseHMM
from surgphase.temporal.smoothing import (
    exponential_moving_average,
    moving_average,
    moving_median,
)

WINDOWED_METHODS = ("moving_average", "moving_median", "exponential_moving_average")
HMM_METHODS = ("hmm_filter", "hmm_viterbi")


def decode(
    probabilities: np.ndarray,
    method: str,
    window: int | None = None,
    causal: bool = True,
    hmm: PhaseHMM | None = None,
) -> np.ndarray:
    if method not in TEMPORAL_METHODS:
        raise ValueError(f"method must be one of {TEMPORAL_METHODS}, got {method!r}")
    if method in WINDOWED_METHODS and window is None:
        raise ValueError(f"{method} needs a window")
    if method in HMM_METHODS and hmm is None:
        raise ValueError(f"{method} needs a fitted PhaseHMM")
    if method == "hmm_viterbi" and causal:
        raise ValueError("hmm_viterbi uses the whole sequence; pass causal=False")

    if method == "none":
        smoothed = probabilities
    elif method == "moving_average":
        smoothed = moving_average(probabilities, window, causal)
    elif method == "moving_median":
        smoothed = moving_median(probabilities, window, causal)
    elif method == "exponential_moving_average":
        if not causal:
            raise ValueError("exponential_moving_average is causal only")
        smoothed = exponential_moving_average(probabilities, window)
    elif method == "hmm_filter":
        smoothed = hmm.filter(probabilities)
    else:
        return hmm.viterbi(probabilities)
    return smoothed.argmax(axis=1)


@dataclass(frozen=True)
class Decoding:
    method: str
    window_seconds: float | None = None
    causal: bool = True

    def window(self, fps: float) -> int | None:
        if self.window_seconds is None:
            return None
        return max(1, round(self.window_seconds * fps))

    def apply(
        self, probabilities: np.ndarray, fps: float, hmm: PhaseHMM | None = None
    ) -> np.ndarray:
        return decode(probabilities, self.method, self.window(fps), self.causal, hmm)


def decodings(
    methods: Sequence[str], smoothing_seconds: Sequence[float], centered: bool
) -> list[Decoding]:
    result = []
    for method in methods:
        if method in WINDOWED_METHODS:
            for seconds in smoothing_seconds:
                result.append(Decoding(method, seconds, True))
                if centered and method != "exponential_moving_average":
                    result.append(Decoding(method, seconds, False))
        else:
            result.append(Decoding(method, None, method != "hmm_viterbi"))
    return result
