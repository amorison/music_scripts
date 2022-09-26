from __future__ import annotations

from pathlib import Path
import typing

from pymusic.plotting import SinglePlotFigure, Plot
from matplotlib import colors
import numpy as np

from .array_on_grid import DumpArrayOnGrid
from .derived_fields import FieldGetter, ProfGetter
from .musicdata import MusicData
from .plots import ScalarPlot, SphericalVectorPlot, SameAxesPlot
from .fort_pp import Contour, ContourSphericalPlot, ContourPlot

if typing.TYPE_CHECKING:
    from typing import List

    from pymusic.big_array import BigArray

    from .array_on_grid import ArrayOnGrid
    from .config import Config, Field


def plot_field(dump: DumpArrayOnGrid, conf_field: Field, radii) -> List[Plot]:
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
            costh=conf_field.costh,
            rbounds=(conf_field.rmin, conf_field.rmax),
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
    contours = [
        Contour(
            "rmark",
            np.full_like(theta := dump.grid.theta_grid.cell_centers(), rad),
            theta if not conf_field.costh else np.cos(theta))
        for rad in radii
    ]
    if conf_field.costh:
        plots.extend(ContourPlot(ctr) for ctr in contours)
    else:
        plots.extend(ContourSphericalPlot(ctr) for ctr in contours)
    return plots


def cmd(conf: Config) -> None:
    figdir = Path("figures")
    figdir.mkdir(parents=True, exist_ok=True)

    var = conf.field.plot
    mdat = MusicData(conf.core.path)
    try:
        renv = mdat.prof1d.params["renv"]
        rcore = mdat.prof1d.params["rcore"]
    except RuntimeError:
        renv = 0.0
        rcore = 0.0
    rschwarz = [rad for rad in (renv, rcore) if rad > 0.0]

    @FieldGetter.register
    def temp(aog: ArrayOnGrid) -> BigArray:
        """Temperature."""
        return mdat.eos.temperature(aog.data)

    for snap in mdat[conf.core.dumps]:
        plots = plot_field(snap.dump_arr, conf.field, radii=rschwarz)
        SinglePlotFigure(
            plot=SameAxesPlot(plots=plots, legend=False),
        ).save_to(figdir / f"{var}_{snap.idump:08d}.png")
