from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import typing

import h5py
import numpy as np
import matplotlib.pyplot as plt
from music_pykg.prof1d import Prof1d

if typing.TYPE_CHECKING:
    from matplotlib.axes import Axes


@dataclass(frozen=True)
class ProfData:
    label: str
    rad: np.ndarray
    pressure: np.ndarray
    temperature: np.ndarray

    def pert_to(self, other: ProfData) -> ProfData:
        """Responsibility of caller that rads are the same."""
        return ProfData(
            rad=self.rad,
            label=f"({self.label} - {other.label}) / {other.label}",
            pressure=(self.pressure - other.pressure) / other.pressure,
            temperature=(self.temperature -
                         other.temperature) / other.temperature,
        )


def get_chk_profile(h5file: str, idump: int, label: str) -> ProfData:
    with h5py.File(h5file) as h5f:
        chkp = h5f["checkpoints"][f"{idump:05d}"]
        data = ProfData(
            pressure=chkp["Moment_rad"]["P_vol"][()].squeeze(),
            temperature=chkp["Moment_rad"]["T_vol"][()].squeeze(),
            rad=chkp["pp_parameters"]["eval_grid"]["rad"][()].squeeze(),
            # time=chkp["parameters"]["time"][()].item(),
            label=label,
        )
    return data


def get_prof1d_at(profile1d: Prof1d, rad: np.ndarray) -> ProfData:
    profs = profile1d.profs
    radc = profs["radc"]
    return ProfData(
        label="1D",
        rad=rad,
        pressure=np.interp(rad, radc[:-1], profs["P"].values[:-1]),
        temperature=np.interp(rad, radc[:-1], profs["T"].values[:-1]))


def draw_prof_on(
    prof: ProfData, profile1d: Prof1d, axis: Axes, xlbl: bool = True
) -> None:
    rad_adim = prof.rad / profile1d.params["rad_surf"]
    axis.plot(rad_adim, prof.pressure, label="Pressure")
    axis.plot(rad_adim, prof.temperature, label="Temperature")
    axis.set_ylabel(prof.label)
    rcore = profile1d.params["rcore"] / profile1d.params["rad_surf"]
    axis.axvline(rcore, lw=1, ls=":", color="k")
    if xlbl:
        axis.set_xlabel("rad/rtot")
    axis.legend()


if __name__ == "__main__":
    prof0 = get_chk_profile("post_transient.h5", 0, "t0")
    prof1 = get_chk_profile("post_transient.h5", 1, "t1")
    proff = get_chk_profile("post_es.h5", 7874, "tf")
    profile1d = Prof1d.with_path_hint(Path())
    prof1ds = get_prof1d_at(profile1d, prof0.rad)

    fig, axis = plt.subplots(nrows=5, sharex=True, figsize=(6, 14))
    draw_prof_on(prof0.pert_to(prof1ds), profile1d, axis[0], False)
    draw_prof_on(prof1.pert_to(prof1ds), profile1d, axis[1], False)
    draw_prof_on(prof1.pert_to(prof0), profile1d, axis[2], False)
    draw_prof_on(proff.pert_to(prof0), profile1d, axis[3], False)
    draw_prof_on(proff.pert_to(prof1), profile1d, axis[4])
    plt.tight_layout()
    plt.savefig("P_T_pert.pdf")
