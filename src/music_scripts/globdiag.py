from __future__ import annotations

from typing import Union

import numpy as np

from .musicdata import MusicData, _SnapsView


def tau_conv(mdat: Union[MusicData, _SnapsView]) -> float:
    """Convective time scale."""
    r_grid = mdat.grid.grids[0]
    d_rad = r_grid.cell_widths()
    core_mask = r_grid.cell_centers() < mdat.prof1d.params["rcore"]
    return (
        (
            mdat.rprof["vrms"].collapse(
                lambda vrms: np.sum(d_rad[core_mask] / vrms[core_mask]), axis="x1"
            )
        )
        .array()
        .mean()
    )
