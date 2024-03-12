from __future__ import annotations

import typing
from pathlib import Path

import numpy as np
from pyevtk.hl import gridToVTK

from .musicdata import MusicData

if typing.TYPE_CHECKING:
    from typing import Iterable

    from .config import Config
    from .musicdata import Snap


def music_to_vtk(
    snaps: Iterable[Snap],
    vtk_path: Path,
    extra_vars: Iterable[str],
) -> None:
    """Convert music files to vtk, assuming constant grid."""
    vtk_path.mkdir(parents=True)

    # grid information
    first_snap = next(iter(snaps))
    all_vars = set(first_snap.big_array.labels_along_axis("var"))
    all_vars.update(extra_vars)

    x1 = first_snap.grid.grids[0].cell_centers()
    x2 = first_snap.grid.grids[1].cell_centers()
    x3 = (
        first_snap.grid.grids[2].cell_centers()
        if first_snap.grid.ndim == 3
        else np.zeros(1)
    )
    if not first_snap.cartesian:
        # spherical
        # projection to cartesian around z axis
        r = x1[:, np.newaxis, np.newaxis]
        t = x2[np.newaxis, :, np.newaxis]
        p = x3[np.newaxis, np.newaxis, :]
        st = np.sin(t)
        sp = np.sin(p)
        ct = np.cos(t)
        cp = np.cos(p)
        st_cp = st * cp
        st_sp = st * sp
        ct_cp = ct * cp
        ct_sp = ct * sp

        xpts = r * st_cp
        ypts = r * st_sp
        zpts = np.ascontiguousarray(np.broadcast_to(r * ct, xpts.shape))

    for snap in snaps:
        vtk_file = vtk_path / f"music{snap.idump:08d}"
        print(f"writing: {vtk_file}")
        flds = {var: np.atleast_3d(snap.field[var].array()) for var in all_vars}  # type: ignore
        if first_snap.cartesian:
            v_3 = flds.pop("vel_3") if "vel_3" in flds else np.zeros_like(flds["vel_1"])
            flds["vel_vec"] = (flds.pop("vel_1"), flds.pop("vel_2"), v_3)  # type: ignore
            gridToVTK(str(vtk_file), x1, x2, x3, pointData=flds)
        else:
            ur = flds["vel_1"]
            ut = flds["vel_2"]
            up = flds["vel_3"] if "vel_3" in flds else np.zeros_like(ur)

            flds["vel_vec"] = (  # type: ignore
                np.ascontiguousarray(ur * st_cp + ut * ct_cp - up * sp),  # ux
                np.ascontiguousarray(ur * st_sp + ut * ct_sp + up * cp),  # uy
                np.ascontiguousarray(
                    np.broadcast_to(ur * ct - ut * st, xpts.shape)
                ),  # uz
            )

            gridToVTK(str(vtk_file), xpts, ypts, zpts, pointData=flds)


def cmd(conf: Config) -> None:
    mdat = MusicData(conf.core.path)
    music_to_vtk(
        mdat[conf.core.dumps],
        conf.vtk.vtk_dir,
        conf.vtk.extra_vars,
    )
