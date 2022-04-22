from __future__ import annotations

import typing
from pathlib import Path

from loam.cli import Subcmd, CLIManager
from loam.manager import ConfOpt as Conf
from loam.manager import ConfigurationManager
from loam.tools import command_flag
import loam.types

from . import field, restart, plot_pendepth, fort_pp

if typing.TYPE_CHECKING:
    from typing import Optional, List


CONF_DEF = {}

CONF_DEF["core"] = dict(
    path=Conf(default=Path(), cmd_arg=True, shortname="p",
              cmd_kwargs={"type": Path}, help="path of music directory",
              comprule="_files"),
)

CONF_DEF["field"] = dict(
    plot=Conf(default="vel_ampl", cmd_arg=True, shortname="o",
              help="variable to plot"),
    velarrow=command_flag(shortname=None,
                          help_msg="add velocity arrows to the plot"),
)

CONF_DEF["restart"] = dict(
    batch=Conf(default=None, cmd_arg=True, shortname="b",
               cmd_kwargs={"nargs": "+"},
               help="batch files to use for restart", comprule="_files")
)

CONF_DEF["fort_pp"] = dict(
    postfile=Conf(default="post.h5", cmd_arg=True, shortname="p",
                  help="path to master h5 file from post_par"),
    idump=Conf(default=1, cmd_arg=True, shortname="d",
               help="dump number to process"),
)

CONF_DEF["field_pp"] = dict(
    plot=Conf(default="rho", cmd_arg=True, shortname="o",
              help="variable to plot"),
)

CONF_DEF["contour_pp"] = dict(
    plot=Conf(default="pen_depth_conv,pen_depth_ke,r_schwarz_max",
              cmd_arg=True, shortname="o",
              cmd_kwargs=dict(type=loam.types.list_of(str)),
              help="variables to plot"),
    over=Conf(default="", cmd_arg=True,
              help="plot the contour over a field variable"),
    rmarks=Conf(default="", cmd_arg=True,
                cmd_kwargs=dict(type=loam.types.list_of(float)),
                help="add contours at constant values"),
)

SUB_CMDS = dict(
    field=Subcmd("plot a scalar field", "core", func=field.cmd),
    restart=Subcmd("restart a MUSIC run from batch file", func=restart.cmd),
    pendepth=Subcmd("plot penetration depth", func=plot_pendepth.cmd),
    field_pp=Subcmd("plot a field from PP data", "fort_pp",
                    func=fort_pp.field_cmd),
    contour_pp=Subcmd("plot a contour field from PP data", "fort_pp",
                      func=fort_pp.contour_cmd),
)


def parse_args_and_run(
    arglist: Optional[List[str]] = None,
) -> None:
    """Parse command line argument, run requested command."""
    conf = ConfigurationManager.from_dict_(CONF_DEF)
    climan = CLIManager(conf, **SUB_CMDS)
    cmd_args = climan.parse_args(arglist)
    cmd_args.func(conf)
