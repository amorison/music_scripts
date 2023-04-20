from __future__ import annotations

import typing

from .globdiag import tau_conv
from .musicdata import MusicData

if typing.TYPE_CHECKING:
    from .config import Config


def cmd(conf: Config) -> None:
    mdat = MusicData(conf.core.path)
    print("Run in:", mdat.path)
    if conf.info.tconv:
        tconv = tau_conv(mdat)
        print("Convective timescale:", tconv)
