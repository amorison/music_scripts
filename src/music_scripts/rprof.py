from __future__ import annotations

import typing

from .figure import SinglePlotFigure
from .musicdata import MusicData
from .plots import ProfPlot, WithScales

if typing.TYPE_CHECKING:
    from .config import Config


def cmd(conf: Config) -> None:
    conf.core.figdir.mkdir(parents=True, exist_ok=True)

    var = conf.rprof.plot
    mdat = MusicData(conf.core.path)

    try:
        rtot = mdat.prof1d.params["rad_surf"]
        renv = mdat.prof1d.params["renv"]
        rcore = mdat.prof1d.params["rcore"]
    except RuntimeError:
        rtot = None
        renv = 0.0
        rcore = 0.0
    rschwarz = [rad for rad in (renv, rcore) if rad > 0.0]
    if conf.plotting.no_rmarks:
        rschwarz = []

    for snap in mdat[conf.core.dumps]:
        plot = ProfPlot(
            music_data=snap,
            var=var,
            markers=rschwarz,
            length_scale=rtot,
        )
        SinglePlotFigure(
            plot=WithScales(
                plot=plot,
                yscale="log" if conf.plotting.log else "linear",
            ),
        ).save_to(conf.core.figdir / f"prof_{var}_{snap.idump:08d}.pdf")
