#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from pymusic.plotting import (
    FfmpegMp4Movie, SinglePlotFigure, Plot, WithPlotTitle,
)
from pymusic.io import MusicSim, PeriodicArrayBC, MusicDumpInfo

from array_on_grid import DumpArrayOnGrid
from derived_fields import FieldGetter
from plots import SphericalPlot


def all_plots(sim: MusicSim, var: str) -> Iterator[Plot]:
    for dump in sim.dumps:
        time = dump.time
        yield WithPlotTitle(
            plot=SphericalPlot(
                dump_arr=DumpArrayOnGrid(dump),
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
    try:
        movie.render_to(f"{var}.mp4")
    except FileNotFoundError as err:
        print(f"Error rendering movie: {err}")


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
