"""One look for every figure: Inter (bundled in docs/fonts, OFL), a small muted palette, light axes.

    from diary_dialogues import figstyle as fs
    fs.apply()                      # once, before plotting
    ax.plot(..., color=fs.APP["phi-4"])
    fs.save(fig, path)

FIGSTYLE=dark switches to a palette for the website's dark mode, and save() then writes name-dark.png next to
name.png. The Markdown shows each pair with GitHub's #gh-light-mode-only / #gh-dark-mode-only markers, which the
MkDocs site also understands.
"""
import os
from pathlib import Path

from diary_dialogues.paths import ROOT

FONT_DIR = ROOT / "docs" / "fonts"

DARK = os.environ.get("FIGSTYLE") == "dark"
if not DARK:
    INK = "#1f2937"        # text
    MUTED = "#6b7280"      # secondary text, axis labels
    LINE = "#d1d5db"       # axes, rules
    GRID = "#eef0f3"
    PAPER = "#ffffff"
    SOFT = "#f6f7f9"       # panels, cards
else:                      # matched to the site's dark ("slate") background
    INK = "#e5e7eb"
    MUTED = "#9ca3af"
    LINE = "#4b5263"
    GRID = "#2b2e3a"
    PAPER = "#1e2029"
    SOFT = "#272a36"

# Fixed colors per app and per profile, the same in every figure.
APP = {"gemma-3-12b": "#3b82c4", "llama-3.1-8b": "#8467c9", "phi-4": "#2a9d8f", "claude-opus-5.5": "#d4793a",
       "claude-haiku-4.5": "#e0a370", "baseline": "#a3a9b3"}
APP_LABEL = {"gemma-3-12b": "Gemma-3-12B", "llama-3.1-8b": "Llama-3.1-8B", "phi-4": "Phi-4",
             "claude-opus-5.5": "Claude Opus 5.5", "claude-haiku-4.5": "Claude Haiku 4.5", "baseline": "Baseline"}
PROFILE = {"control": "#a3a9b3", "Social_Isolation": "#3b82c4", "Entitlement": "#d4793a",
           "Unrelenting_Standards": "#2a9d8f", "Enmeshment": "#8467c9"}
PROFILE_LABEL = {"control": "Unsteered", "Social_Isolation": "Social isolation", "Entitlement": "Entitlement",
                 "Unrelenting_Standards": "Unrelenting standards", "Enmeshment": "Enmeshment"}
ACCENT = "#3b82c4"


def _cmap(colors):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("figstyle", colors)


CMAP = "Blues" if not DARK else _cmap([PAPER, "#25466b", ACCENT, "#9cc7f0"])            # sequential
DIVERGING = "RdBu_r" if not DARK else _cmap(["#4a8fd4", "#2d4566", SOFT, "#6e3b2c", "#d4793a"])


def apply():
    import matplotlib as mpl
    from matplotlib import font_manager
    for f in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(f))
    mpl.rcParams.update({
        "font.family": "Inter", "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "semibold",
        "axes.titlelocation": "left", "axes.titlepad": 10, "axes.labelsize": 9, "axes.labelcolor": MUTED,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.edgecolor": LINE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 5, "ytick.major.pad": 5,
        "legend.frameon": False, "legend.fontsize": 8.5, "figure.facecolor": PAPER, "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER, "lines.linewidth": 1.8, "lines.markersize": 4.5,
        "figure.titlesize": 11.5, "figure.titleweight": "semibold",
    })


def heading(fig, title: str, note: str | None = None, y: float | None = None, note_y: float = -0.01):
    """Every figure: a short title on top, an optional muted source note at the bottom, both flush with the
    leftmost content (aligned in save). By default the title sits a fixed distance above the highest axes."""
    from matplotlib.transforms import offset_copy
    if y is None:
        top = max(ax.get_position().y1 for ax in fig.axes)
        pts = 30 if any(ax.get_title(loc=l) for ax in fig.axes for l in ("left", "center")) else 10
        t = fig.text(0.01, top, title, ha="left", va="bottom", fontsize=11.5, weight="semibold", color=INK,
                     transform=offset_copy(fig.transFigure, fig=fig, y=pts, units="points"))
    else:
        t = fig.text(0.01, y, title, ha="left", va="bottom", fontsize=11.5, weight="semibold", color=INK)
    fig._fs_texts = [t]
    if note:
        fig._fs_texts.append(fig.text(0.01, note_y, note, ha="left", va="top", fontsize=8, color=MUTED))


def save(fig, path: Path):
    import matplotlib.pyplot as plt
    texts = getattr(fig, "_fs_texts", [])
    if texts:                                    # align heading and note with the leftmost drawn content
        for t in texts:
            t.set_visible(False)
        r = fig.canvas.get_renderer()
        x0 = fig.get_tightbbox(r).x0 / fig.get_figwidth()
        for t in texts:
            t.set_visible(True)
            t.set_x(x0)
    path = Path(path)
    if DARK:
        path = path.with_name(path.stem + "-dark" + path.suffix)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
