from __future__ import annotations

from pathlib import Path
import typing

from pymusic.plotting import SinglePlotFigure, Plot
from matplotlib import colors
import numpy as np

from .derived_fields import FieldGetter, ProfGetter
from .musicdata import MusicData
from .plots import ScalarPlot, SphericalVectorPlot, SameAxesPlot
from .fort_pp import Contour, ContourSphericalPlot, ContourPlot

if typing.TYPE_CHECKING:
    from typing import Sequence, Callable, List

    from pymusic.big_array import BigArray

    from .config import Config, Field
    from .musicdata import BaseMusicData, Snap


def plot_field(
    snap: Snap,
    conf_field: Field,
) -> Sequence[Plot]:
    var = conf_field.plot
    cmap = conf_field.cmap

    field_getter: Callable[[BaseMusicData], BigArray]
    if conf_field.perturbation:
        def field_getter(bmdat: BaseMusicData) -> BigArray:
            """Temperature perturbation."""
            field = FieldGetter(var)(bmdat)
            prof = ProfGetter(var)(bmdat).array()
            return field.apply(lambda f: (f - prof[:, np.newaxis]) / f)
        if cmap is None:
            cmap = "RdBu_r"
    else:
        field_getter = FieldGetter(var)

    normr = not conf_field.full_r
    mdat = snap.mdat

    try:
        rtot = mdat.prof1d.params["rad_surf"] if normr else None
        renv = mdat.prof1d.params["renv/rtot" if normr else "renv"]
        rcore = mdat.prof1d.params["rcore/rtot" if normr else "rcore"]
    except RuntimeError:
        rtot = None
        renv = 0.0
        rcore = 0.0
    rschwarz = [rad for rad in (renv, rcore) if rad > 0.0]

    plots: List[Plot] = [
        ScalarPlot(
            dump_arr=snap,
            get_data=field_getter,
            cmap=cmap,
            norm=(None if not conf_field.perturbation
                  else colors.SymLogNorm(linthresh=1e-6)),
            costh=conf_field.costh,
            rbounds=(conf_field.rmin, conf_field.rmax),
            vbounds=(conf_field.vmin, conf_field.vmax),
            normalize_r=rtot,
        ),
    ]
    if conf_field.velarrow:
        plots.append(
            SphericalVectorPlot(
                dump_arr=snap,
                get_rvec=FieldGetter("vel_1"),
                get_tvec=FieldGetter("vel_2"),
            )
        )
    contours = [
        Contour(
            "rschwarz",
            np.full_like(theta := snap.grid.theta_grid.cell_centers(), rad),
            theta if not conf_field.costh else np.cos(theta))
        for rad in rschwarz
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

    for snap in mdat[conf.core.dumps]:
        plots = plot_field(snap, conf.field)
        SinglePlotFigure(
            plot=SameAxesPlot(plots=plots, legend=False),
        ).save_to(figdir / f"{var}_{snap.idump:08d}.png")
