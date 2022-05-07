from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

from loam.base import entry, Section, ConfigBase
from loam.cli import Subcmd, CLIManager
from loam.collections import TupleEntry
from loam.tools import command_flag, path_entry

from . import field, restart, plot_pendepth, fort_pp, lmax


@dataclass
class Core(Section):
    path: Path = path_entry(path=".", cli_short="P",
                            doc="path of music directory")


@dataclass
class Field(Section):
    plot: str = entry(val="vel_ampl", cli_short="o", doc="variable to plot")
    velarrow: bool = command_flag("add velocity arrows to the plot")


@dataclass
class Restart(Section):
    batch: Tuple[str, ...] = TupleEntry(str).entry(
        doc="batch files to use for restart", cli_short="b",
        cli_zsh_comprule="_files")


@dataclass
class FortPP(Section):
    postfile: Path = path_entry(path="post.h5", cli_short="p",
                                doc="path to master h5 file from post_par")
    idump: int = entry(val=1, cli_short="d", doc="dump number to process")


@dataclass
class Plotting(Section):
    rmarks: Tuple[float, ...] = TupleEntry(float).entry(
        doc="add contours at constant values")


@dataclass
class FieldPP(Section):
    plot: str = entry(val="rho", cli_short="o", doc="variable to plot")


@dataclass
class ContourPP(Section):
    plot: Tuple[str, ...] = TupleEntry(str).entry(
        default="pen_depth_conv,pen_depth_ke,r_schwarz_max",
        cli_short="o", doc="variables to plot")
    over: str = entry(val="", doc="plot the contour over a field variable")


@dataclass
class RprofPP(Section):
    plot: str = entry(val="rho", cli_short="o", doc="variable to plot")
    degree: int = entry(val=1, cli_short="D", doc="degree of rprof")


@dataclass
class Lmax(Section):
    normdr: bool = command_flag("normalize lmax by average dr")


@dataclass
class Config(ConfigBase):
    core: Core
    field: Field
    restart: Restart
    fort_pp: FortPP
    plotting: Plotting
    field_pp: FieldPP
    contour_pp: ContourPP
    rprof_pp: RprofPP
    lmax: Lmax


SUB_CMDS = dict(
    field=Subcmd("plot a scalar field", "core", func=field.cmd),
    restart=Subcmd("restart a MUSIC run from batch file", func=restart.cmd),
    pendepth=Subcmd("plot penetration depth", func=plot_pendepth.cmd),
    field_pp=Subcmd("plot a field from PP data",
                    "fort_pp", "plotting",
                    func=fort_pp.field_cmd),
    contour_pp=Subcmd("plot a contour field from PP data",
                      "fort_pp", "plotting",
                      func=fort_pp.contour_cmd),
    rprof_pp=Subcmd("plot a rprof field from PP data",
                    "fort_pp", "plotting",
                    func=fort_pp.rprof_cmd),
    lmax=Subcmd("plot lmax histogram", "fort_pp", func=lmax.cmd)
)


def parse_args_and_run(
    arglist: Optional[List[str]] = None,
) -> None:
    """Parse command line argument, run requested command."""
    conf = Config.default_()
    climan = CLIManager(conf, **SUB_CMDS)
    cmd_args = climan.parse_args(arglist)
    cmd_args.func(conf)
