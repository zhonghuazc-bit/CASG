#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 1 (v3): detailed CASG workflow, English version for the SR submission.

Style/granularity follows the Chinese reference flowchart:
two preset boxes, S1-S5 boxes with 3 detail lines each, a convergence diamond,
a dashed Brent-search feedback loop, an output box, and a legend band.

All text and numbers are taken from Section 3 of main.tex.
Outputs:
  投稿/sr/LaTeX_submission/casg_framework.png  (raster used by pdflatex)
  投稿/sr/LaTeX_submission/casg_framework.svg  (vector source)
  paper/figures/casg_framework.png             (working copy)
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle

ROOT = Path(r'C:\Users\zc\Documents\周翀\研究课题\LLM辅助教育')
OUT_DIR = ROOT / '投稿' / 'sr' / 'LaTeX_submission'
WORK_DIR = ROOT / 'paper' / 'figures'

# ---- palette (light theme) -------------------------------------------------
BLUE_F, BLUE_E = '#EAF3FC', '#5A8DB8'
AMBER_F, AMBER_E = '#FFF6E2', '#D6A23C'
GREEN_F, GREEN_E = '#E7F4E9', '#4F9A63'
ORANGE_F, ORANGE_E = '#FFF4EA', '#DD7A38'
LEG_F, LEG_E = '#F7F8FA', '#D8DDE3'
LOOP = '#C0392B'
TEXT, SUB = '#22272C', '#3E454B'
ARROW = '#4A5057'
YES = '#2E7D46'

# ---- shared typography: one sans-serif face for all four figures ----------
from fig_font import apply_figure_font

apply_figure_font()

fig, ax = plt.subplots(figsize=(8.0, 10.0), dpi=300)
ax.set_xlim(0, 8)
ax.set_ylim(0, 10)
ax.axis('off')
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# ---- helpers ---------------------------------------------------------------
def rbox(x, y, w, h, fc, ec, lw=1.3):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.012,rounding_size=0.075',
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)
    return p


def down_arrow(x, y1, y2, color=ARROW, lw=1.4):
    a = FancyArrowPatch((x, y1), (x, y2), arrowstyle='-|>',
                        mutation_scale=13, lw=lw, color=color,
                        shrinkA=0, shrinkB=0, zorder=3)
    ax.add_patch(a)


def stage(y, h, header, lines, fc, ec, header_fs=10.4, body_fs=8.9,
          header_color=TEXT):
    """Box with a centred bold header and centred detail lines."""
    x, w = 1.45, 5.10
    rbox(x, y, w, h, fc, ec)
    cx = x + w / 2
    ax.text(cx, y + 0.80 * h, header, ha='center', va='center',
            fontsize=header_fs, fontweight='bold', color=header_color, zorder=4)
    n = len(lines)
    if n == 3:
        ys = [y + 0.615 * h, y + 0.445 * h, y + 0.275 * h]
    elif n == 2:
        ys = [y + 0.56 * h, y + 0.32 * h]
    else:
        ys = [y + 0.44 * h]
    for yy, line in zip(ys, lines):
        ax.text(cx, yy, line, ha='center', va='center',
                fontsize=body_fs, color=SUB, zorder=4)


def preset(x, w, y, h, header, lines):
    rbox(x, y, w, h, BLUE_F, BLUE_E)
    cx = x + w / 2
    ax.text(cx, y + h - 0.16, header, ha='center', va='center',
            fontsize=10.2, fontweight='bold', color=TEXT, zorder=4)
    ys = [y + h - 0.34, y + h - 0.48, y + h - 0.62]
    for yy, line in zip(ys, lines):
        ax.text(cx, yy, line, ha='center', va='center',
                fontsize=8.15, color=SUB, zorder=4)


# ---- title -----------------------------------------------------------------
ax.text(4.0, 9.80,
        'Coverage-Aware Structured Grading (CASG): Overall Workflow',
        ha='center', va='center', fontsize=13.4, fontweight='bold',
        color=TEXT, zorder=4)

# ---- two preset boxes ------------------------------------------------------
preset(0.30, 3.55, 8.86, 0.70, 'Inputs', [
    'question texts Q · student answers A',
    'institutional distribution targets',
    'M questions, N students',
])
preset(4.15, 3.55, 8.86, 0.70, 'Preset Parameters', [
    r'target difficulty $P_t$ · discrimination $D_t$ · std. dev. $\sigma_t$',
    r'loss weights $w_1 = w_2 = 1, w_3 = 0.1$',
    r'$0 \leq \alpha \leq 2$, Brent tolerance 1e-6',
])

# bracket joining the two preset boxes into the main flow
ax.plot([2.075, 2.075, 4.0], [8.86, 8.68, 8.68], color=ARROW, lw=1.2, zorder=1)
ax.plot([5.925, 5.925, 4.0], [8.86, 8.68, 8.68], color=ARROW, lw=1.2, zorder=1)
down_arrow(4.0, 8.68, 8.56)

# ---- S1 S2 -----------------------------------------------------------------
stage(7.51, 1.05, 'S1 · Question Decomposition (LLM)', [
    'Decompose each question into K scoring points, each into L sub-points',
    'Assign max_score to each scoring point; no human rubric engineering',
    'Output: scoring-point set KP, sub-point set SP, max scores',
], AMBER_F, AMBER_E)
down_arrow(4.0, 7.51, 7.37)

stage(6.32, 1.05, 'S2 · Coverage Assessment (LLM)', [
    'Classify status: full / partial / missing for each scoring point',
    'coverage_ratio = |covered sub-points| / |total sub-points|',
    'Output: status flags, ratios, evidence excerpts, reasoning',
], AMBER_F, AMBER_E)
down_arrow(4.0, 6.32, 6.18)

# ---- S3 core ---------------------------------------------------------------
stage(5.13, 1.05, 'S3 · Power-Function Coverage Mapping (Core)', [
    r'full: max_score · partial: max_score × coverage_ratio^$\alpha$',
    'missing: 0.05 × max_score, minimal participation credit',
    r'$\alpha < 1$ gives a concave curve, lenient to partial understanding',
], GREEN_F, GREEN_E, header_color=GREEN_E)
down_arrow(4.0, 5.13, 4.99)

# ---- S4 --------------------------------------------------------------------
stage(3.94, 1.05, 'S4 · Hierarchical Aggregation', [
    r'question score = $\Sigma$ scoring-point scores',
    r'total score = $\Sigma$ (question score × normalized weight), range 0–100',
], AMBER_F, AMBER_E)
down_arrow(4.0, 3.94, 3.80)

# ---- convergence diamond ---------------------------------------------------
diamond = Polygon([(4.0, 3.80), (6.15, 3.44), (4.0, 3.08), (1.85, 3.44)],
                  closed=True, fc=ORANGE_F, ec=ORANGE_E, lw=1.4, zorder=2)
ax.add_patch(diamond)
ax.text(4.0, 3.52, r'Has $L(\alpha)$ converged?', ha='center', va='center',
        fontsize=10.0, fontweight='bold', color=TEXT, zorder=4)
ax.text(4.0, 3.35, 'weighted relative-deviation loss minimized',
        ha='center', va='center', fontsize=7.5, color=SUB, zorder=4)

# ---- dashed feedback loop --------------------------------------------------
dash = (0, (4, 3))
ax.plot([6.15, 7.45], [3.44, 3.44], color=LOOP, lw=1.3,
        linestyle=dash, zorder=1)
ax.plot([7.45, 7.45], [3.44, 4.465], color=LOOP, lw=1.3,
        linestyle=dash, zorder=1)
back = FancyArrowPatch((7.45, 4.465), (6.56, 4.465), arrowstyle='-|>',
                       mutation_scale=12, lw=1.3, color=LOOP,
                       linestyle=dash, shrinkA=0, shrinkB=0, zorder=3)
ax.add_patch(back)
ax.text(7.35, 3.70, r'No: update $\alpha$', ha='right', va='center',
        fontsize=8.3, color=LOOP, fontweight='bold', zorder=4)
ax.text(7.35, 3.56, 'bounded Brent search', ha='right', va='center',
        fontsize=7.4, color=LOOP, zorder=4)

# ---- yes branch ------------------------------------------------------------
ax.text(4.20, 3.00, 'Yes', ha='left', va='center',
        fontsize=8.5, color=YES, fontweight='bold', zorder=4)
down_arrow(4.0, 3.08, 2.94)

# ---- S5 core ---------------------------------------------------------------
stage(1.89, 1.05, 'S5 · Distribution-Target-Driven Exponent Selection (Core)', [
    r'Select $\alpha^*$ minimizing $L(\alpha)$ over $0 \leq \alpha \leq 2$ by Brent’s method',
    r'Report $P$, $D$, $\sigma$ against targets; $\alpha$ is dataset-specific',
    'Anchors difficulty while preserving human-score agreement',
], GREEN_F, GREEN_E, header_fs=9.9, header_color=GREEN_E)
down_arrow(4.0, 1.89, 1.75)

# ---- output ----------------------------------------------------------------
stage(0.70, 1.05, 'Output', [
    'per-student scoring-point scores, question scores, total score',
    'coverage diagnostics: covered / missed sub-points, evidence, rationale',
    r'distribution report: $P$, $D$, $\sigma$ and deviations from targets',
], BLUE_F, BLUE_E, header_fs=10.6)
# output box has no downward arrow; nothing to add

# ---- legend band -----------------------------------------------------------
rbox(0.30, 0.04, 7.40, 0.60, LEG_F, LEG_E, lw=1.0)
ax.plot([4.70, 4.70], [0.12, 0.56], color=LEG_E, lw=0.9, zorder=3)


def swatch(x, y, fc, ec, label, dashed=False):
    if dashed:
        ax.plot([x, x + 0.22], [y, y], color=ec, lw=1.8,
                linestyle=dash, zorder=4)
    else:
        ax.add_patch(Rectangle((x, y - 0.065), 0.16, 0.13,
                               fc=fc, ec=ec, lw=1.0, zorder=4))
    ax.text(x + 0.30, y, label, ha='left', va='center',
            fontsize=8.0, color=SUB, zorder=4)


swatch(0.48, 0.46, GREEN_F, GREEN_E, 'Core contribution')
swatch(0.48, 0.20, AMBER_F, AMBER_E, 'LLM-involved stage')
swatch(2.60, 0.46, ORANGE_F, ORANGE_E, 'Convergence decision')
swatch(2.60, 0.20, None, LOOP, 'Iterative optimization loop', dashed=True)

ax.text(4.88, 0.48, r'$0 \leq \alpha \leq 2$ · Brent tolerance 1e-6',
        ha='left', va='center', fontsize=7.9, color=SUB, zorder=4)
ax.text(4.88, 0.30, r'defaults: $w_1 = w_2 = 1, w_3 = 0.1$',
        ha='left', va='center', fontsize=7.9, color=SUB, zorder=4)
ax.text(4.88, 0.12, 'missing-status floor: 0.05 × max_score',
        ha='left', va='center', fontsize=7.9, color=SUB, zorder=4)

# ---- export ----------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
png = OUT_DIR / 'casg_framework.png'
svg = OUT_DIR / 'casg_framework.svg'
fig.savefig(png, dpi=300, facecolor='white')
fig.savefig(svg, facecolor='white')
fig.savefig(WORK_DIR / 'casg_framework.png', dpi=300, facecolor='white')
fig.savefig(WORK_DIR / 'casg_framework.svg', facecolor='white')
plt.close(fig)
print('written:', png)
print('written:', svg)
print('written:', WORK_DIR / 'casg_framework.png')
