import numpy as np

from .musicdata import MusicData


def tau_conv(mdat: MusicData) -> float:
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
