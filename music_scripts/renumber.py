from __future__ import annotations

from pathlib import Path
import typing

if typing.TYPE_CHECKING:
    from .config import Config


def rename_in(
    folder_in: Path, folder_out: Path, ifirst: int, pattern_out: str = "{:08}.music",
) -> None:
    folder_out.mkdir()
    for i, filepath in enumerate(sorted(folder_in.glob("*.music")), ifirst):
        newpath = folder_out / pattern_out.format(i)
        filepath.rename(newpath)


def cmd(conf: Config) -> None:
    rename_in(conf.renumber.path_in, conf.renumber.path_out, conf.renumber.ifirst)
