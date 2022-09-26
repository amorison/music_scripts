from __future__ import annotations

from pathlib import Path
import typing

from pymusic.plotting import Figure, SinglePlotFigure
from matplotlib import colors
import numpy as np

from .array_on_grid import DumpArrayOnGrid
from .derived_fields import FieldGetter, ProfGetter
from .musicdata import MusicData
from .plots import ScalarPlot, SphericalVectorPlot, SameAxesPlot

if typing.TYPE_CHECKING:
    from pymusic.big_array import BigArray

    from .array_on_grid import ArrayOnGrid
    from .config import Config, Field


def plot_field(dump: DumpArrayOnGrid, conf_field: Field) -> Figure:
    var = conf_field.plot
    cmap = conf_field.cmap
    if conf_field.perturbation:
        def field_getter(aog: ArrayOnGrid) -> BigArray:
            """Temperature perturbation."""
            field = FieldGetter(var)(aog)
            prof = ProfGetter(var)(aog).array()
            return field.apply(lambda f: (f - prof[:, np.newaxis]) / f)
        if cmap is None:
            cmap = "RdBu_r"
    else:
        field_getter = FieldGetter(var)

    plots = [
        ScalarPlot(
            dump_arr=dump,
            get_data=field_getter,
            cmap=cmap,
            norm=(None if not conf_field.perturbation
                  else colors.SymLogNorm(linthresh=1e-6)),
        ),
    ]
    if conf_field.velarrow:
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

    @FieldGetter.register
    def temp(aog: ArrayOnGrid) -> BigArray:
        """Temperature."""
        return mdat.eos.temperature(aog.data)

    for snap in mdat[conf.core.dumps]:
        fig = plot_field(snap.dump_arr, conf.field)
        fig.save_to(figdir / f"{var}_{snap.idump:08d}.png")
