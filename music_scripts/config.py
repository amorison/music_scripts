from __future__ import annotations

import typing
from pathlib import Path

from loam.cli import Subcmd, CLIManager
from loam.manager import ConfOpt as Conf
from loam.manager import ConfigurationManager
from loam.tools import command_flag

from . import field, restart, plot_pendepth

if typing.TYPE_CHECKING:
    from typing import Optional, List, Callable


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

SUB_CMDS = dict(
    field=Subcmd("plot a scalar field", "core", func=field.cmd),
    restart=Subcmd("restart a MUSIC run from batch file", func=restart.cmd),
    pendepth=Subcmd("plot penetration depth", func=plot_pendepth.cmd),
)


def parse_args_and_run(
    arglist: Optional[List[str]] = None,
) -> None:
    """Parse command line argument, run requested command."""
    conf = ConfigurationManager.from_dict_(CONF_DEF)
    climan = CLIManager(conf, **SUB_CMDS)
    cmd_args = climan.parse_args(arglist)
    cmd_args.func(conf)
