from __future__ import annotations

import shlex
import subprocess
import typing
from pathlib import Path

import f90nml

if typing.TYPE_CHECKING:
    from .config import Config


def restart_batch(batchfile: Path) -> None:
    """Restart MUSIC run using info in batchfile."""
    content = batchfile.read_text()
    content_parts = content.strip().split()

    old_music_out = content_parts[-2]
    out_number = int(old_music_out[-6:-4])
    new_music_out = old_music_out[:-6] + f"{out_number+1:02d}.out"
    print(f"{batchfile}: {old_music_out} > {new_music_out}")

    params = Path(content_parts[-5])
    nml = f90nml.read(str(params))
    old_input = nml["io"]["input"]

    output = nml["io"]["dataoutput"]
    new_input = str(max(Path().glob(f"{output}*.music")))

    print(f"{params}: {old_input} > {new_input}")

    confirm = input("Confirm (y/N)? ")
    if confirm.lower() != "y":
        return

    content = content.replace(old_music_out, new_music_out, 1)
    batchfile.write_text(content)
    params_content = params.read_text()
    params_content = params_content.replace(old_input, new_input, 1)
    params.write_text(params_content)
    subprocess.run(shlex.split(f"sbatch '{batchfile}'"), check=True)


def cmd(conf: Config) -> None:
    if not conf.restart.batch:
        batch_files = tuple(Path().glob("batch*"))
    else:
        batch_files = tuple(map(Path, conf.restart.batch))
    for batch in batch_files:
        restart_batch(batch)
