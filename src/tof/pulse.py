# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipp as sc

from . import facilities
from .utils import Plot, wavelength_to_speed


class Pulse:
    """
    A class that represents a pulse of neutrons.
    It is defined by the number of neutrons, a wavelength range, and a time range.
    A probability distribution for the wavelengths and times can be provided (the
    distribution is flat by default).
    In addition, some pre-defined pulses from neutron facilities can be used by setting
    the ``kind`` parameter.
    Finally, the wavelengths and times of the neutrons can be provided directly.

    Parameters
    ----------
    tmin:
        Start time of the pulse.
    tmax:
        End time of the pulse.
    wmin:
        Minimum wavelength of the pulse.
    wmax:
        Maximum wavelength of the pulse.
    neutrons:
        Number of neutrons in the pulse.
    kind:
        Name of a pre-defined pulse from a neutron facility.
    p_wav:
        Probability distribution for the wavelengths.
    p_time:
        Probability distribution for the times.
    sampling_resolution:
        Number of points used to sample the probability distributions.
    birth_times:
        Birth times of neutrons in the pulse.
    wavelengths:
        Wavelengths of neutrons in the pulse.
    """

    def __init__(
        self,
        tmin: Optional[sc.Variable] = None,
        tmax: Optional[sc.Variable] = None,
        wmin: Optional[sc.Variable] = None,
        wmax: Optional[sc.Variable] = None,
        neutrons: int = 1_000_000,
        generate: bool = True,
    ):
        self.birth_times = None
        self.wavelengths = None
        self.speeds = None
        self.kind = None

        self.neutrons = neutrons
        self.tmin = tmin
        self.tmax = tmax
        self.wmin = wmin
        self.wmax = wmax

        if generate:
            self.birth_times = sc.array(
                dims=['event'],
                values=np.random.uniform(
                    self.tmin.value, self.tmax.value, self.neutrons
                ),
                unit='s',
            )
            self.wavelengths = sc.array(
                dims=['event'],
                values=np.random.uniform(
                    self.wmin.value, self.wmax.value, self.neutrons
                ),
                unit='angstrom',
            )
            self.speeds = wavelength_to_speed(self.wavelengths)

    @classmethod
    def from_neutrons(cls, birth_times: sc.Variable, wavelengths: sc.Variable):
        """
        Create a pulse from a list of neutrons.
        Both ``birth times`` and ``wavelengths`` should be one-dimensional, and have the
        same length and dimension label.

        Parameters
        ----------
        birth_times:
            Birth times of neutrons in the pulse.
        wavelengths:
            Wavelengths of neutrons in the pulse.
        """
        pulse = cls(generate=False)
        pulse.birth_times = birth_times.to(unit='s')
        pulse.wavelengths = wavelengths.to(unit='angstrom')
        pulse.tmin = pulse.birth_times.min()
        pulse.tmax = pulse.birth_times.max()
        pulse.wmin = pulse.wavelengths.min()
        pulse.wmax = pulse.wavelengths.max()
        pulse.speeds = wavelength_to_speed(pulse.wavelengths)
        pulse.neutrons = len(pulse.birth_times)
        return pulse

    @classmethod
    def from_facility(
        cls,
        kind: str,
        tmin: Optional[sc.Variable] = None,
        tmax: Optional[sc.Variable] = None,
        wmin: Optional[sc.Variable] = None,
        wmax: Optional[sc.Variable] = None,
        neutrons: int = 1_000_000,
        sampling: int = 100_000,
    ):
        """
        Create a pulse from a pre-defined pulse from a neutron facility.

        Parameters
        ----------
        kind:
            Name of a pre-defined pulse from a neutron facility.
        tmin:
            Start time of the pulse.
        tmax:
            End time of the pulse.
        wmin:
            Minimum wavelength of the pulse.
        wmax:
            Maximum wavelength of the pulse.
        neutrons:
            Number of neutrons in the pulse.
        sampling:
            Number of points used to sample the probability distributions.
        """
        pulse = cls(
            tmin=tmin,
            tmax=tmax,
            wmin=wmin,
            wmax=wmax,
            neutrons=neutrons,
            generate=False,
        )
        pulse.kind = kind
        params = getattr(facilities, pulse.kind)
        if pulse.tmin is None:
            pulse.tmin = params.time.coords['time'].min()
        if pulse.tmax is None:
            pulse.tmax = params.time.coords['time'].max()
        if pulse.wmin is None:
            pulse.wmin = params.wavelength.coords['wavelength'].min()
        if pulse.wmax is None:
            pulse.wmax = params.wavelength.coords['wavelength'].max()
        pulse.tmin = pulse.tmin.to(unit='s')
        pulse.tmax = pulse.tmax.to(unit='s')
        pulse.wmin = pulse.wmin.to(unit='angstrom')
        pulse.wmax = pulse.wmax.to(unit='angstrom')

        sampling = int(sampling)
        x_time = np.linspace(pulse.tmin.value, pulse.tmax.value, sampling)
        x_wav = np.linspace(pulse.wmin.value, pulse.wmax.value, sampling)
        p_time = np.interp(
            x_time,
            params.time.coords['time'].to(unit='s').values,
            params.time.values,
        )
        p_time /= p_time.sum()
        p_wav = np.interp(
            x_wav,
            params.wavelength.coords['wavelength'].to(unit='angstrom').values,
            params.wavelength.values,
        )
        p_wav /= p_wav.sum()

        pulse.birth_times = sc.array(
            dims=['event'],
            values=np.random.choice(x_time, size=pulse.neutrons, p=p_time),
            unit='s',
        )
        pulse.wavelengths = sc.array(
            dims=['event'],
            values=np.random.choice(x_wav, size=pulse.neutrons, p=p_wav),
            unit='angstrom',
        )
        pulse.speeds = wavelength_to_speed(pulse.wavelengths)
        return pulse

    @classmethod
    def from_distribution(
        cls,
        p_time: sc.DataArray,
        p_wav: sc.DataArray,
        neutrons: int = 1_000_000,
        sampling: Optional[int] = None,
    ):
        """
        Create a pulse from time a wavelength probability distributions.
        The distributions should be supplied as DataArrays where the coordinates
        are the values of the distribution, and the values are the probability.
        Note that the time and wavelength distributions are independent. A neutron with
        a randomly selected birth time from ``p_time`` can adopt any wavelength in
        ``p_wav`` (in other words, the two distributions are simply broadcast into a
        square 2D parameter space).

        Parameters
        ----------
        p_time:
            Time probability distribution.
        p_wav:
            Wavelength probability distribution.
        neutrons:
            Number of neutrons in the pulse.
        sampling:
            Number of points used to sample the probability distributions. If not set,
            the size of the distributions will be used.
        """
        pulse = cls(
            tmin=p_time.coords['time'].min().to(unit='s'),
            tmax=p_time.coords['time'].max().to(unit='s'),
            wmin=p_wav.coords['wavelength'].min().to(unit='angstrom'),
            wmax=p_wav.coords['wavelength'].max().to(unit='angstrom'),
            neutrons=neutrons,
            generate=False,
        )
        p_time = p_time.to(dtype=float)
        p_wav = p_wav.to(dtype=float)

        if sampling is None:
            x_time = p_time.coords['time'].to(dtype=float, unit='s').values
            p_time = p_time.values
            x_wav = p_wav.coords['wavelength'].to(dtype=float, unit='angstrom').values
            p_wav = p_wav.values
        else:
            sampling = int(sampling)
            x_time = np.linspace(pulse.tmin.value, pulse.tmax.value, sampling)
            x_wav = np.linspace(pulse.wmin.value, pulse.wmax.value, sampling)
            p_time = np.interp(
                x_time,
                p_time.coords['time'].to(unit='s').values,
                p_time.values,
            )
            p_wav = np.interp(
                x_wav,
                p_wav.coords['wavelength'].to(unit='angstrom').values,
                p_wav.values,
            )

        p_time = p_time / p_time.sum()
        p_wav = p_wav / p_wav.sum()

        pulse.birth_times = sc.array(
            dims=['event'],
            values=np.random.choice(x_time, size=pulse.neutrons, p=p_time),
            unit='s',
        )
        pulse.wavelengths = sc.array(
            dims=['event'],
            values=np.random.choice(x_wav, size=pulse.neutrons, p=p_wav),
            unit='angstrom',
        )
        pulse.speeds = wavelength_to_speed(pulse.wavelengths)
        return pulse

    @property
    def duration(self) -> float:
        """Duration of the pulse."""
        return self.tmax - self.tmin

    def plot(self, bins: int = 300) -> tuple:
        """
        Plot the pulse.

        Parameters
        ----------
        bins:
            Number of bins to use for histogramming the neutrons.
        """
        fig, ax = plt.subplots(1, 2)
        self.birth_times.hist(time=bins).plot(ax=ax[0])
        self.wavelengths.hist(wavelength=bins).plot(ax=ax[1])
        size = fig.get_size_inches()
        fig.set_size_inches(size[0] * 2, size[1])
        fig.tight_layout()
        return Plot(fig=fig, ax=ax)

    def as_readonly(self):
        return ReadonlyPulse(
            birth_times=self.birth_times,
            wavelengths=self.wavelengths,
            speeds=self.speeds,
            kind=self.kind,
            neutrons=self.neutrons,
            tmin=self.tmin,
            tmax=self.tmax,
            wmin=self.wmin,
            wmax=self.wmax,
        )

    def __repr__(self) -> str:
        return (
            f"Pulse(tmin={self.tmin:c}, tmax={self.tmax:c}, "
            f"wmin={self.wmin:c}, wmax={self.wmax:c}, "
            f"neutrons={self.neutrons}, kind={self.kind})"
        )


@dataclass(frozen=True)
class ReadonlyPulse:
    birth_times: sc.DataArray
    wavelengths: sc.DataArray
    speeds: sc.DataArray
    kind: Optional[str]
    neutrons: int
    tmin: sc.Variable
    tmax: sc.Variable
    wmin: sc.Variable
    wmax: sc.Variable
