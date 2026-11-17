from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from sklearn.metrics import ConfusionMatrixDisplay

from surgphase.config import UNLABELLED
from surgphase.viz.style import INK, SEQUENTIAL, SURFACE, style_axes


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    phase_names: Sequence[str],
    ax: Axes | None = None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 6.5), layout="constrained")
    labelled = labels != UNLABELLED
    display = ConfusionMatrixDisplay.from_predictions(
        labels[labelled],
        predictions[labelled],
        labels=list(range(len(phase_names))),
        display_labels=list(phase_names),
        normalize="true",
        cmap=SEQUENTIAL,
        values_format=".2f",
        xticks_rotation=45,
        colorbar=False,
        ax=ax,
        im_kw={"vmin": 0, "vmax": 1},
        text_kw={"fontsize": 8},
    )
    style_axes(ax)
    for text in display.text_.ravel():
        if text.get_text() == "0.00":
            text.set_text("")
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    edges = np.arange(len(phase_names) + 1) - 0.5
    ax.set_xticks(edges, minor=True)
    ax.set_yticks(edges, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("Predicted phase")
    ax.set_ylabel("True phase")
    ax.title.set_color(INK)
    return ax
