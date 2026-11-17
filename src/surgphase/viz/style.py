from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
NEUTRAL = "#c3c2b7"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "surgphase_blue",
    ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)


def phase_colors(num_phases: int, neutral_phase: int | None) -> list[str]:
    needed = num_phases - (neutral_phase is not None)
    if needed > len(CATEGORICAL):
        raise ValueError(
            f"{needed} phases need distinct hues but only {len(CATEGORICAL)} are available"
        )
    hues = iter(CATEGORICAL)
    return [NEUTRAL if index == neutral_phase else next(hues) for index in range(num_phases)]


def style_axes(ax: Axes) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED_INK, labelcolor=SECONDARY_INK)
    ax.xaxis.label.set_color(SECONDARY_INK)
    ax.yaxis.label.set_color(SECONDARY_INK)
    ax.title.set_color(INK)
