# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc

from . import facilities
from .units import units


class Pulse:
    def __init__(
        self,
        tmin: float = Optional[sc.Variable],
        tmax: float = Optional[sc.Variable],
        lmin: float = Optional[sc.Variable],
        lmax: float = Optional[sc.Variable],
        neutrons: int = 1_000_000,
        kind: str = Optional[None],
        p_wav: np.ndarray = Optional[None],
        p_time: np.ndarray = Optional[None],
        sampling_resolution: int = 10000,
    ):
        self.kind = kind
        self.neutrons = neutrons

        if self.kind is not None:
            params = getattr(facilities, self.kind)
            self.tmin = params['time'].coords['time'].min().to(unit='s')
            self.tmax = params['time'].coords['time'].max().to(unit='s')
            self.lmin = (
                params['wavelength'].coords['wavelength'].min().to(unit='angstrom')
            )
            self.lmax = (
                params['wavelength'].coords['wavelength'].max().to(unit='angstrom')
            )

            x_time = np.linspace(self.tmin.value, self.tmax.value, sampling_resolution)
            x_wav = np.linspace(self.lmin.value, self.lmax.value, sampling_resolution)
            p_time = np.interp(
                x_time,
                params['time'].coords['time'].to(unit='s').values,
                params['time'].values,
            )
            p_time /= p_time.sum()
            p_wav = np.interp(
                x_wav,
                params['wavelength'].coords['wavelength'].to(unit='angstrom').values,
                params['wavelength'].values,
            )
            p_wav /= p_wav.sum()
        else:
            self.tmin = tmin.to(unit='s')
            self.tmax = tmax.to(unit='s')
            self.lmin = lmin.to(unit='angstrom')
            self.lmax = lmax.to(unit='angstrom')

        if p_time is not None:
            if self.kind is None:
                x_time = np.linspace(self.tmin.value, self.tmax.value, len(p_time))
            self.birth_times = np.random.choice(x_time, size=self.neutrons, p=p_time)
        else:
            self.birth_times = np.random.uniform(
                self.tmin.value, self.tmax.value, self.neutrons
            )
        if p_wav is not None:
            if self.kind is None:
                x_wav = np.linspace(self.lmin.value, self.lmax.value, len(p_wav))
            self.wavelengths = np.random.choice(x_wav, size=self.neutrons, p=p_wav)
        else:
            self.wavelengths = np.random.uniform(self.lmin, self.lmax, self.neutrons)

        self.birth_times = sc.array(dims=['event'], values=self.birth_times, unit='s')
        self.wavelengths = sc.array(
            dims=['event'], values=self.wavelengths, unit='angstrom'
        )
        self.speeds = units.wavelength_to_speed(self.wavelengths)

    @property
    def duration(self) -> float:
        return self.tmax - self.tmin

    def plot(self, bins: int = 300) -> tuple:
        fig, ax = plt.subplots(1, 2)
        self.birth_times.hist(time=bins).plot(ax=ax[0])
        self.wavelengths.hist(wavelength=bins).plot(ax=ax[1])
        size = fig.get_size_inches()
        fig.set_size_inches(size[0] * 2, size[1])
        fig.tight_layout()
        return fig, ax

    def __repr__(self) -> str:
        return (
            f"Pulse(tmin={self.tmin:c}, tmax={self.tmax:c}, lmin={self.lmin:c}, "
            f"lmax={self.lmax:c}, neutrons={self.neutrons}, kind={self.kind})"
        )
