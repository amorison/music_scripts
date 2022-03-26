"""Utility to read profile1d dat files."""
from __future__ import annotations
import typing
from pathlib import Path
from types import MappingProxyType

import pandas as pd

if typing.TYPE_CHECKING:
    from os import PathLike
    from typing import Union, Optional, Mapping, Dict


class Prof1d:

    """Prof1d parser.

    Args:
        path_hint: either the path to the profile file, or the path to the
            folder containing the profile file.  In the latter case, the
            parser tries to find a file and fails if none is found or there
            is ambiguity.
    """

    def __init__(self, path_hint: Union[str, PathLike]):
        self._path_hint = Path(path_hint)
        self._path: Optional[Path] = None
        self._params: Optional[Dict[str, float]] = None
        self._profs: Optional[pd.DataFrame] = None

    @property
    def path(self) -> Path:
        """Path to the profile file."""
        if self._path is not None:
            return self._path
        if self._path_hint.is_file():
            self._path = self._path_hint
            return self._path
        for p1d in ("profile1d.dat", "profile1d_scalars.dat"):
            file_path = self._path_hint / p1d
            if file_path.exists():
                if self._path is not None:
                    raise RuntimeError(
                        "Two profile1d files found: {file_path} and "
                        "{self._path}")
                self._path = file_path
        if self._path is None:
            raise RuntimeError("No profile1d file found in {self._path_hint}")
        return self._path

    @property
    def params(self) -> Mapping[str, float]:
        if self._params is not None:
            return MappingProxyType(self._params)
        with self.path.open() as p1d:
            names = p1d.readline().split()
            values = map(float, p1d.readline().split())
        self._params = dict(zip(names, values))
        return self.params

    @property
    def profs(self) -> pd.DataFrame:
        if self._profs is not None:
            return self._profs
        self._profs = pd.read_csv(self.path, skiprows=2, delim_whitespace=True)
        return self._profs


if __name__ == "__main__":
    prof1d = Prof1d(".")
    profs = prof1d.profs
    import matplotlib.pyplot as plt
    rtot = prof1d.params["rad_surf"]
    plt.semilogy(profs["r_grid"]/rtot, profs["Enuc"])
    plt.axvline(prof1d.params["rcore/rtot"], lw=1, color='k', ls=":")
    plt.xlabel("r/rtot")
    plt.ylabel("Enuc")
    plt.savefig("enuc_prof.pdf")
