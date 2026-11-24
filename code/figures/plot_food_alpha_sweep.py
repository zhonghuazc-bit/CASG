"""Generate Fig 4 (combined, two side-by-side panels) for the Food Engineering
cross-disciplinary experiment.

Left panel (a) = pure short-answer formulation; right panel (b) = total-score formulation.
Each panel is a dual-axis chart merging Loss, QWK10, Pearson r and MAE over the full alpha
sweep. Design is grayscale-safe (unique line style + unique marker shape + distinct gray
level per metric). The right-hand (log) axis uses plain decimal tick labels - 2, 3, 4, 6,
10, 20, 30 - with no scientific notation.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from fig_font import apply_figure_font

apply_figure_font()

D = json.load(open("casg_alpha_sweep_table.json", encoding="utf-8"))

# metric -> (key, label, gray level, linestyle, marker, filled?)
STYLE = {
    "loss": ("loss_pdsigma01", "Loss L",    "#000000", "-",  "o", True),
    "qwk":  ("qwk_10",         "QWK10",     "#404040", "--", "s", False),
    "r":    ("pearson_r",      "Pearson r", "#6E6E6E", "-.", "^", True),
    "mae":  ("mae",            "MAE",       "#949494", ":",  "D", False),
}

# right axis: plain decimal ticks only, no scientific notation
RIGHT_TICKS = [2, 3, 4, 6, 10, 20, 30]
RIGHT_LABELS = [f"{t:g}" for t in RIGHT_TICKS]

PANELS = [
    ("(a)  Pure short-answer formulation", D["short_sweep"], (0.28, 1.03), (1.5, 45.0)),
    ("(b)  Total-score formulation",       D["total_sweep"], (0.84, 1.005), (1.15, 8.0)),
]

fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.6))
handles, labels = [], []

for i, (title, sweep, ylim_L, ylim_R) in enumerate(PANELS):
    axL = axes[i]
    axR = axL.twinx()
    al = np.array([r["alpha"] for r in sweep])
    val = lambda k, s=sweep: np.array([r[STYLE[k][0]] for r in s])

    hmap = {}
    for k in ("qwk", "r"):                      # left axis (0-1 metrics)
        _, lab, col, ls, mk, filled = STYLE[k]
        h, = axL.plot(al, val(k), color=col, ls=ls, lw=2.2, marker=mk, ms=5.0,
                      mfc=(col if filled else "white"), mec=col, mew=1.4, label=lab)
        hmap[k] = h
    for k in ("loss", "mae"):                   # right axis (log metrics)
        _, lab, col, ls, mk, filled = STYLE[k]
        h, = axR.plot(al, val(k), color=col, ls=ls, lw=2.2, marker=mk, ms=5.0,
                      mfc=(col if filled else "white"), mec=col, mew=1.4, label=lab)
        hmap[k] = h
    if i == 0:
        legend_order = ["loss", "qwk", "r", "mae"]   # legend: loss, qwk, r, mae
        handles = [hmap[k] for k in legend_order]
        labels = [STYLE[k][1] for k in legend_order]

    # ---- left axis ----
    axL.set_xlim(-0.06, 2.06)
    axL.set_ylim(*ylim_L)
    axL.set_xlabel(r"power exponent $\alpha$", fontsize=11)
    axL.set_ylabel("QWK10  /  Pearson r", fontsize=10.5)
    axL.grid(True, ls=":", color="#CCCCCC", alpha=0.9)
    axL.set_axisbelow(True)

    # ---- right axis: log scale, PLAIN integer labels only ----
    axR.set_yscale("log")
    axR.set_ylim(*ylim_R)
    axR.yaxis.set_major_locator(FixedLocator(RIGHT_TICKS))
    axR.yaxis.set_major_formatter(FixedFormatter(RIGHT_LABELS))
    axR.yaxis.set_minor_locator(NullLocator())
    axR.set_ylabel("Loss L  /  MAE", fontsize=10.5)

    # ---- optimum alpha* = 0 (label rotated inside the band to avoid any overlap) ----
    axL.axvspan(-0.06, 0.035, color="#D9D9D9", alpha=0.85, zorder=0)
    axL.text(-0.0125, 0.5 * (ylim_L[0] + ylim_L[1]), r"$\alpha^* = 0$",
             fontsize=10.5, color="#000000", fontweight="bold",
             rotation=90, va="center", ha="center", zorder=3)

    axL.set_title(title, fontsize=11.5, fontweight="bold", pad=8)

fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9.5,
           frameon=True, framealpha=0.95, edgecolor="#999999",
           handlelength=2.8, columnspacing=2.0, bbox_to_anchor=(0.5, -0.015))

fig.tight_layout(rect=[0, 0.055, 1, 1])
fig.savefig("food_alpha_sweep.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("saved food_alpha_sweep.png (combined 2-panel figure, plain right-axis ticks)")
