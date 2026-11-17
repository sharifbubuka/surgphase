from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from surgphase.config import UNLABELLED
from surgphase.eval.metrics import segments
from surgphase.viz.style import GRIDLINE, INK, SECONDARY_INK, style_axes


def plot_timeline(
    rows: Mapping[str, np.ndarray],
    phase_names: Sequence[str],
    colors: Sequence[str],
    fps: float,
    ax: Axes | None = None,
) -> Axes:
    if len(colors) != len(phase_names):
        raise ValueError(f"{len(phase_names)} phases but {len(colors)} colors")
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 1.6 + 0.45 * len(rows)), layout="constrained")
    style_axes(ax)
    names = list(rows)
    duration = 0.0
    for position, name in enumerate(names):
        y = len(names) - 1 - position
        labels, starts, ends = segments(np.asarray(rows[name]))
        for phase in np.unique(labels):
            if phase == UNLABELLED:
                continue
            chosen = labels == phase
            spans = [
                (start / fps / 60, (end - start) / fps / 60)
                for start, end in zip(starts[chosen], ends[chosen], strict=True)
            ]
            ax.broken_barh(spans, (y - 0.35, 0.7), facecolors=colors[phase], linewidth=0)
        duration = max(duration, len(rows[name]) / fps / 60)
    ax.set_yticks(range(len(names)), labels=reversed(names))
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.set_xlim(0, duration)
    ax.set_xlabel("Time (min)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)
    present = sorted({int(p) for row in rows.values() for p in np.unique(row) if p != UNLABELLED})
    legend = ax.figure.legend(
        handles=[Patch(facecolor=colors[p], label=phase_names[p]) for p in present],
        loc="outside lower center",
        ncol=min(4, len(present)),
        frameon=False,
        handlelength=1.0,
        handleheight=1.0,
    )
    for text in legend.get_texts():
        text.set_color(SECONDARY_INK)
    return ax
