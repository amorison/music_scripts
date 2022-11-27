from __future__ import annotations

import typing

from .figure import SinglePlotFigure
from .musicdata import MusicData
from .plots import TseriesPlot, WithScales

if typing.TYPE_CHECKING:
    from .config import Config


def cmd(conf: Config) -> None:
    conf.core.figdir.mkdir(parents=True, exist_ok=True)

    var = conf.tseries.plot
    mdat = MusicData(conf.core.path)

    SinglePlotFigure(
        plot=WithScales(
            plot=TseriesPlot(mdat, var),
            yscale="log" if conf.plotting.log else "linear",
        ),
    ).save_to(conf.core.figdir / f"tseries_{var}.pdf")
