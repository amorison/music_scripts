#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from pymusic.plotting import (
    FfmpegMp4Movie, SinglePlotFigure, Plot, WithPlotTitle,
)

from .musicdata import MusicData
from .derived_fields import FieldGetter
from .plots import ScalarPlot


def all_plots(mdat: MusicData, var: str) -> Iterator[Plot]:
    for snap in mdat:
        time = snap.dump.time
        yield WithPlotTitle(
            plot=ScalarPlot(
                dump_arr=snap,
                get_data=FieldGetter(var),
            ),
            title=f"{var} at time {time:.2e}"
        )


def main(mdat: MusicData, var: str) -> None:
    movie = FfmpegMp4Movie(
        figures=tuple(
            SinglePlotFigure(plot) for plot in all_plots(mdat, var)
        ),
        frames_dir=Path(f"frames_{var}"),
    )
    try:
        movie.render_to(f"{var}.mp4")
    except FileNotFoundError as err:
        print(f"Error rendering movie: {err}")


if __name__ == "__main__":
    main(MusicData(Path("params.nml")), "vel_ampl")
