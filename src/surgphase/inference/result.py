from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionResult:
    predictions: np.ndarray
    confidences: np.ndarray
