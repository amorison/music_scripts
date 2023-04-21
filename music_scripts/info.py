from __future__ import annotations

import typing

from .globdiag import tau_conv
from .musicdata import MusicData

if typing.TYPE_CHECKING:
    from typing import Union

    from .config import Config
    from .musicdata import _SnapsView


def cmd(conf: Config) -> None:
    mdat = MusicData(conf.core.path)
    print("Run in:", mdat.path)
    view: Union[MusicData, _SnapsView] = (
        mdat if conf.core.dumps == () else mdat[conf.core.dumps]
    )
    if conf.info.tconv:
        tconv = tau_conv(view)
        print(f"Convective timescale: {tconv:e}")
