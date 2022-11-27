from __future__ import annotations

import typing

from .figure import SinglePlotFigure
from .musicdata import MusicData
from .plots import TseriesPlot

if typing.TYPE_CHECKING:
    from .config import Config


def cmd(conf: Config) -> None:
    conf.core.figdir.mkdir(parents=True, exist_ok=True)

    var = conf.tseries.plot
    mdat = MusicData(conf.core.path)

    SinglePlotFigure(
        plot=TseriesPlot(mdat, var),
    ).save_to(conf.core.figdir / f"tseries_{var}.pdf")
