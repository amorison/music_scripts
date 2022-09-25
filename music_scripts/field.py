from __future__ import annotations

import typing

from .plots_old import plot_var
from .musicdata import MusicData

if typing.TYPE_CHECKING:
    from .config import Config


def cmd(conf: Config) -> None:
    mdat = MusicData(conf.core.path)
    plot_var(mdat.sim_arr_on_grid, conf.field.plot, conf.field.velarrow)
