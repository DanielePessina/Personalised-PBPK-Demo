"""Notebook-local Matplotlib styling helpers."""

from __future__ import annotations

from importlib import resources
from typing import Final, Literal

from cycler import cycler
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import seaborn as sns


FontName = Literal["Roboto Slab", "Paper Mono"]

_STYLE_PACKAGE: Final[str] = "notebook_support.mpl_styles"
_FONT_FILES: Final[dict[FontName, str]] = {
    "Roboto Slab": "RobotoSlab-VariableFont_wght.ttf",
    "Paper Mono": "PaperMono-Regular.ttf",
}
_REGISTERED_FONTS: set[str] = set()


def use_mpl_style(font: FontName = "Roboto Slab") -> None:
    """Register vendored fonts and apply the notebook plotting style.

    Parameters
    ----------
    font : {"Roboto Slab", "Paper Mono"}, optional
        Font family to use for notebook plots.

    Returns
    -------
    None
    """
    _register_font("Roboto Slab")
    _register_font(font)

    style_resource = resources.files(_STYLE_PACKAGE).joinpath("plot.mplstyle")
    with resources.as_file(style_resource) as style_path:
        plt.style.use(style_path)

    dark2_palette = sns.color_palette("Dark2", n_colors=8)
    mpl.rcParams["font.family"] = font
    mpl.rcParams["axes.prop_cycle"] = cycler(color=dark2_palette)
    sns.set_palette(dark2_palette)


def _register_font(font: FontName) -> None:
    """Register a vendored font with Matplotlib once per session."""
    font_resource = resources.files(_STYLE_PACKAGE).joinpath(_FONT_FILES[font])
    with resources.as_file(font_resource) as font_path:
        font_path_str = str(font_path)
        if font_path_str in _REGISTERED_FONTS:
            return

        font_manager.fontManager.addfont(font_path)
        _REGISTERED_FONTS.add(font_path_str)
