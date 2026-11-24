"""Shared typography for the four Scientific Reports figures.

Scientific Reports Figure guidelines require a clear sans-serif typeface used
consistently across all figures in a paper, with a symbol font for Greek
letters. Helvetica is not installed on this Windows machine, so Arial is used,
which is the metrically compatible alternative named in Nature's figure
guidelines. Matplotlib mathtext supplies the symbol face for Greek letters and
mathematical notation.

Usage:  from fig_font import apply_figure_font; apply_figure_font()
"""

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager

_FONT_DIR = Path(r'C:\Windows\Fonts')
_ARIAL_FACES = ('arial.ttf', 'arialbd.ttf', 'ariali.ttf', 'arialbi.ttf')


def apply_figure_font():
    """Apply one typeface and one size system before any figure is drawn."""
    for face in _ARIAL_FACES:
        font_manager.fontManager.addfont(str(_FONT_DIR / face))

    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.sans-serif'] = ['Arial']
    mpl.rcParams['font.serif'] = ['Arial']
    mpl.rcParams['mathtext.fontset'] = 'dejavusans'
    mpl.rcParams['svg.fonttype'] = 'path'

    # One size system shared by the flow chart and the three curve figures.
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.titlesize'] = 11
    mpl.rcParams['axes.labelsize'] = 10.5
    mpl.rcParams['xtick.labelsize'] = 9
    mpl.rcParams['ytick.labelsize'] = 9
    mpl.rcParams['legend.fontsize'] = 9
