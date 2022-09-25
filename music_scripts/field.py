from __future__ import annotations

from pathlib import Path
import typing

from pymusic.plotting import Figure, SinglePlotFigure

from .array_on_grid import DumpArrayOnGrid
from .derived_fields import FieldGetter
from .musicdata import MusicData
from .plots import ScalarPlot, SphericalVectorPlot, SameAxesPlot

if typing.TYPE_CHECKING:
    from .config import Config


def plot_field(
    dump: DumpArrayOnGrid, var: str, vel_arrows: bool = False,
) -> Figure:
    plots = [
        ScalarPlot(
            dump_arr=dump,
            get_data=FieldGetter(var),
        ),
    ]
    if vel_arrows:
        plots.append(
            SphericalVectorPlot(
                dump_arr=dump,
                get_rvec=FieldGetter("vel_1"),
                get_tvec=FieldGetter("vel_2"),
            )
        )
    return SinglePlotFigure(
        plot=SameAxesPlot(plots=plots, legend=False),
    )


def cmd(conf: Config) -> None:
    figdir = Path("figures")
    figdir.mkdir(parents=True, exist_ok=True)

    var = conf.field.plot
    mdat = MusicData(conf.core.path)
    for snap in mdat[conf.core.dumps]:
        fig = plot_field(snap.dump_arr, var, conf.field.velarrow)
        fig.save_to(figdir / f"{var}_{snap.idump:08d}.png")
