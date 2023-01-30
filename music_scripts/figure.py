from __future__ import annotations

import typing
from dataclasses import dataclass

import matplotlib.figure as mplf
from pymusic.plotting import Figure, Plot

if typing.TYPE_CHECKING:
    from typing import Tuple


@dataclass(frozen=True)
class SinglePlotFigure(Figure):
    plot: Plot
    figsize: Tuple[float, float] = (6.4, 4.8)
    dpi: int = 200

    def save_to(self, file_name: str) -> None:
        fig = mplf.Figure(figsize=self.figsize)
        ax = fig.add_subplot()
        self.plot.draw_on(ax)
        fig.set_tight_layout(True)
        fig.savefig(str(file_name), dpi=self.dpi, bbox_inches="tight")
