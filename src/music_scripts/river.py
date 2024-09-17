from __future__ import annotations

import typing

from .figure import SinglePlotFigure
from .musicdata import MusicData
from .plots import RiverPlot

if typing.TYPE_CHECKING:
    from .config import Config


def cmd(conf: Config) -> None:
    conf.core.figdir.mkdir(parents=True, exist_ok=True)
    var = conf.river.plot
    plot = RiverPlot(
        music_data=MusicData(conf.core.path),
        var=var,
    )
    SinglePlotFigure(
        plot=plot,
    ).save_to(conf.core.figdir / f"river_{var}.pdf")
