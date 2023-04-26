# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc

from . import facilities
from .utils import wavelength_to_speed


class Pulse:
    def __init__(
        self,
        tmin: Optional[sc.Variable] = None,
        tmax: Optional[sc.Variable] = None,
        lmin: Optional[sc.Variable] = None,
        lmax: Optional[sc.Variable] = None,
        neutrons: int = 1_000_000,
        kind: Optional[str] = None,
        p_wav: Optional[np.ndarray] = None,
        p_time: Optional[np.ndarray] = None,
        sampling_resolution: int = 10000,
    ):
        self.kind = kind
        self.neutrons = neutrons
        self.tmin = tmin
        self.tmax = tmax
        self.lmin = lmin
        self.lmax = lmax
        self.p_wav = p_wav
        self.p_time = p_time
        self.sampling_resolution = sampling_resolution
        self.generate()

    def generate(self, neutrons: Optional[int] = None):
        if neutrons is not None:
            self.neutrons = neutrons
        if self.kind is not None:
            params = getattr(facilities, self.kind)
            if self.tmin is None:
                self.tmin = params['time'].coords['time'].min()
            if self.tmax is None:
                self.tmax = params['time'].coords['time'].max()
            if self.lmin is None:
                self.lmin = params['wavelength'].coords['wavelength'].min()
            if self.lmax is None:
                self.lmax = params['wavelength'].coords['wavelength'].max()
            self.tmin = self.tmin.to(unit='s')
            self.tmax = self.tmax.to(unit='s')
            self.lmin = self.lmin.to(unit='angstrom')
            self.lmax = self.lmax.to(unit='angstrom')

            x_time = np.linspace(
                self.tmin.value, self.tmax.value, self.sampling_resolution
            )
            x_wav = np.linspace(
                self.lmin.value, self.lmax.value, self.sampling_resolution
            )
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
            self.tmin = self.tmin.to(unit='s')
            self.tmax = self.tmax.to(unit='s')
            self.lmin = self.lmin.to(unit='angstrom')
            self.lmax = self.lmax.to(unit='angstrom')
            p_time = self.p_time
            p_wav = self.p_wav

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
        self.speeds = wavelength_to_speed(self.wavelengths)

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
