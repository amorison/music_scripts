from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

from loam.base import ConfigBase, Section, entry
from loam.cli import CLIManager, Subcmd
from loam.collections import MaybeEntry, TupleEntry
from loam.parsers import slice_or_int_parser
from loam.tools import command_flag, path_entry

from . import (
    field,
    fort_pp,
    igw,
    info,
    lmax,
    lyon1d,
    plot_pendepth,
    renumber,
    restart,
    river,
    rprof,
    to_vtk,
    tseries,
)

_idx = TupleEntry(slice_or_int_parser)


@dataclass
class Core(Section):
    path: Path = path_entry(
        path="params.nml", cli_short="P", doc="path of music parameter file"
    )
    dumps: Tuple[Union[int, slice], ...] = _idx.entry(
        default=(-1,), doc="sequence of dumps to process", cli_short="d"
    )
    figdir: Path = path_entry(
        path="figures", cli_short="O", doc="folder where produced figures are put"
    )


@dataclass
class Field(Section):
    plot: str = entry(val="vel_ampl", cli_short="o", doc="variable to plot")
    velarrow: bool = command_flag("add velocity arrows to the plot")
    perturbation: bool = command_flag("perturbation")
    cmap: Optional[str] = MaybeEntry(str).entry(doc="matplotlib color map")
    costh: bool = command_flag("plot spherical in (cos th, r) cartesian")
    rmin: Optional[float] = MaybeEntry(float).entry(
        doc="min radius on plot (with costh)"
    )
    rmax: Optional[float] = MaybeEntry(float).entry(
        doc="max radius on plot (with costh)"
    )
    vmin: Optional[float] = MaybeEntry(float).entry(doc="min field value on plot")
    vmax: Optional[float] = MaybeEntry(float).entry(doc="max field value on plot")
    full_r: bool = command_flag("do not try to normalize r by rtot")


@dataclass
class Igw(Section):
    field: str = entry(val="vel_1", doc="field to use to compute spectrum")
    ells: Tuple[int, ...] = TupleEntry(int).entry(
        default=(1, 2, 3, 5, 10, 15),
        doc="angular degrees",
    )


@dataclass
class Tseries(Section):
    plot: str = entry(val="ekin", cli_short="o", doc="variable to plot")


@dataclass
class Prof(Section):
    plot: str = entry(val="ekin", cli_short="o", doc="variable to plot")


@dataclass
class River(Section):
    plot: str = entry(val="ekin", cli_short="o", doc="variable to plot")


@dataclass
class Vtk(Section):
    vtk_dir: Path = path_entry(path="vtk", cli_short="V", doc="output directory")


@dataclass
class Info(Section):
    tconv: bool = command_flag(doc="convective timescale")


@dataclass
class Restart(Section):
    batch: Tuple[str, ...] = TupleEntry(str).entry(
        doc="batch files to use for restart", cli_short="b", cli_zsh_comprule="_files"
    )


@dataclass
class Renumber(Section):
    path_in: Path = path_entry(
        path=".", cli_short="P", doc="directory containing MUSIC files"
    )
    path_out: Path = path_entry(
        path="renumbered", cli_short="O", doc="output directory"
    )
    ifirst: int = entry(val=1, cli_short="i", doc="index to start from")


@dataclass
class FortPP(Section):
    postfile: Path = path_entry(
        path="post.h5", cli_short="p", doc="path to master h5 file from post_par"
    )
    idump: int = entry(val=1, cli_short="i", doc="dump number to process")


@dataclass
class Plotting(Section):
    rmarks: Tuple[float, ...] = TupleEntry(float).entry(
        doc="add contours at constant values"
    )
    log: bool = command_flag("set log scale")
    no_rmarks: bool = command_flag("do not plot radial marks")


@dataclass
class FieldPP(Section):
    plot: str = entry(val="density", cli_short="o", doc="variable to plot")


@dataclass
class ContourPP(Section):
    plot: Tuple[str, ...] = TupleEntry(str).entry(
        default="pen_depth_conv,pen_depth_ke,r_schwarz_max",
        cli_short="o",
        doc="variables to plot",
    )
    over: str = entry(val="", doc="plot the contour over a field variable")


@dataclass
class RprofPP(Section):
    plot: str = entry(val="density", cli_short="o", doc="variable to plot")
    degree: int = entry(val=1, cli_short="D", doc="degree of rprof")


@dataclass
class Lmax(Section):
    normdr: bool = command_flag("normalize lmax by average dr")


@dataclass
class Lyon1d(Section):
    mfile: Path = path_entry("fort50", doc="path to the file to read", cli_short="m")
    plot: Tuple[str, ...] = TupleEntry(str).entry(
        default="temperature", doc="list of variables to plot", cli_short="o"
    )


@dataclass
class Config(ConfigBase):
    core: Core
    field: Field
    igw: Igw
    tseries: Tseries
    rprof: Prof
    river: River
    vtk: Vtk
    info: Info
    restart: Restart
    renumber: Renumber
    fort_pp: FortPP
    plotting: Plotting
    field_pp: FieldPP
    contour_pp: ContourPP
    rprof_pp: RprofPP
    lmax: Lmax
    lyon1d: Lyon1d


SUB_CMDS = dict(
    field=Subcmd("plot a scalar field", "core", "plotting", func=field.cmd),
    igw=Subcmd("compute spectrum", "core", func=igw.cmd),
    tseries=Subcmd("plot a time series", "core", "plotting", func=tseries.cmd),
    rprof=Subcmd("plot a radial profile", "core", "plotting", func=rprof.cmd),
    river=Subcmd("plot a river plot (time, radius)", "core", func=river.cmd),
    vtk=Subcmd("convert music files to VTK", "core", func=to_vtk.cmd),
    info=Subcmd("general info about a run", "core", func=info.cmd, dumps=()),
    restart=Subcmd("restart a MUSIC run from batch file", func=restart.cmd),
    renumber=Subcmd("renumber MUSIC output file", func=renumber.cmd),
    pendepth=Subcmd("plot penetration depth", func=plot_pendepth.cmd),
    field_pp=Subcmd(
        "plot a field from PP data", "fort_pp", "plotting", func=fort_pp.field_cmd
    ),
    contour_pp=Subcmd(
        "plot a contour field from PP data",
        "fort_pp",
        "plotting",
        func=fort_pp.contour_cmd,
    ),
    rprof_pp=Subcmd(
        "plot a rprof field from PP data", "fort_pp", "plotting", func=fort_pp.rprof_cmd
    ),
    lyon1d=Subcmd("plot 1D data from Lyon model", func=lyon1d.cmd),
    lmax=Subcmd("plot lmax histogram", "fort_pp", func=lmax.cmd),
)


def parse_args_and_run(
    arglist: Optional[List[str]] = None,
) -> None:
    """Parse command line argument, run requested command."""
    conf = Config.default_()
    climan = CLIManager(conf, **SUB_CMDS)
    cmd_args = climan.parse_args(arglist)
    cmd_args.func(conf)
