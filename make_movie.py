#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional
from dataclasses import dataclass

import numpy as np
from pymusic.plotting import (
    FfmpegMp4Movie, SinglePlotFigure, Plot, WithPlotTitle,
    BoundsFromMinMax,
)
from pymusic.io import MusicSim, PeriodicArrayBC, MusicDumpInfo, MusicDump

from derived_fields import FieldGetter


@dataclass(frozen=True)
class SphericalPlot(Plot):
    dump: MusicDump
    get_data: FieldGetter
    cmap: Optional[str] = None
    color_bounds = BoundsFromMinMax()
    with_colorbar: bool = True

    def draw_on(self, ax) -> None:
        rad = self.dump.grid.r_grid.face_points()
        theta = self.dump.grid.theta_grid.face_points()
        data = self.get_data(self.dump.big_array()).array()
        vmin, vmax = self.color_bounds(data)
        r_mesh, t_mesh = np.meshgrid(rad, theta, indexing="ij")
        x_mesh = r_mesh * np.cos(t_mesh)
        y_mesh = r_mesh * np.sin(t_mesh)
        surf = ax.pcolormesh(
            x_mesh, y_mesh, data, cmap=self.cmap,
            vmin=vmin, vmax=vmax, shading="flat", rasterized=True)
        ax.set_aspect("equal")
        ax.set_axis_off()
        if self.with_colorbar:
            ax.figure.colorbar(surf, ax=ax)


def all_plots(sim: MusicSim, var: str) -> Iterator[Plot]:
    for dump in sim.dumps:
        time = dump.time
        yield WithPlotTitle(
            plot=SphericalPlot(
                dump=dump,
                get_data=FieldGetter(var),
            ),
            title=f"{var} at time {time:.2e}"
        )


def main(sim, var) -> None:
    movie = FfmpegMp4Movie(
        figures=tuple(
            SinglePlotFigure(plot) for plot in all_plots(sim, var)
        ),
        frames_dir=Path(f"frames_{var}"),
    )
    movie.render_to(f"{var}.mp4")


if __name__ == "__main__":
    dump_dir = Path()
    sim = MusicSim.from_dump_dir(
        directory=str(dump_dir),
        dump_info=MusicDumpInfo(
            num_space_dims=2,
            num_velocities=2,
            num_scalars=1,
        ),
        recenter_bc_list=[PeriodicArrayBC(), PeriodicArrayBC()]
    )
    main(sim, "vel_ampl")
