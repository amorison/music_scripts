from __future__ import annotations

import typing
from pathlib import Path

from loam.cli import Subcmd, CLIManager
from loam.manager import ConfOpt as Conf
from loam.manager import ConfigurationManager
from loam.tools import command_flag

from . import field

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

SUB_CMDS = dict(
    field=Subcmd("plot a scalar field",
                 "core", func=field.cmd),
)


def parse_args(
    arglist: Optional[List[str]] = None,
) -> Callable[[ConfigurationManager], None]:
    """Parse command line argument, return command to run."""
    conf = ConfigurationManager.from_dict_(CONF_DEF)
    climan = CLIManager(conf, **SUB_CMDS)
    cmd_args = climan.parse_args(arglist)
    cmd_args.func(conf)
