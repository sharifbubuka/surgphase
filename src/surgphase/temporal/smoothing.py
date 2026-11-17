import numpy as np
import pandas as pd
from pandas.core.window.rolling import Rolling


def _rolling(probabilities: np.ndarray, window: int, causal: bool) -> Rolling:
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    return pd.DataFrame(probabilities).rolling(window, min_periods=1, center=not causal)


def moving_average(probabilities: np.ndarray, window: int, causal: bool) -> np.ndarray:
    return _rolling(probabilities, window, causal).mean().to_numpy()


def moving_median(probabilities: np.ndarray, window: int, causal: bool) -> np.ndarray:
    smoothed = _rolling(probabilities, window, causal).median().to_numpy()
    return smoothed / smoothed.sum(axis=1, keepdims=True).clip(min=np.finfo(float).tiny)


def exponential_moving_average(probabilities: np.ndarray, span: float) -> np.ndarray:
    if span < 1:
        raise ValueError(f"span must be >= 1, got {span}")
    return pd.DataFrame(probabilities).ewm(span=span, adjust=False).mean().to_numpy()
