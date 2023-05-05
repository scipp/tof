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
    t_min:
        Start time of the pulse.
    t_max:
        End time of the pulse.
    wav_min:
        Minimum wavelength of the pulse.
    wav_max:
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
        t_min: Optional[sc.Variable] = None,
        t_max: Optional[sc.Variable] = None,
        wav_min: Optional[sc.Variable] = None,
        wav_max: Optional[sc.Variable] = None,
        neutrons: int = 1_000_000,
        generate: bool = True,
    ):
        self.birth_times = None
        self.wavelengths = None
        self.speeds = None
        self.kind = None

        self.neutrons = neutrons
        self.t_min = t_min
        self.t_max = t_max
        self.wav_min = wav_min
        self.wav_max = wav_max

        if generate:
            self.birth_times = sc.array(
                dims=['event'],
                values=np.random.uniform(
                    self.t_min.value, self.t_max.value, self.neutrons
                ),
                unit='s',
            )
            self.wavelengths = sc.array(
                dims=['event'],
                values=np.random.uniform(
                    self.wav_min.value, self.wav_max.value, self.neutrons
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
        pulse.t_min = pulse.birth_times.min()
        pulse.t_max = pulse.birth_times.max()
        pulse.wav_min = pulse.wavelengths.min()
        pulse.wav_max = pulse.wavelengths.max()
        pulse.speeds = wavelength_to_speed(pulse.wavelengths)
        pulse.neutrons = len(pulse.birth_times)
        return pulse

    @classmethod
    def from_facility(
        cls,
        kind: str,
        t_min: Optional[sc.Variable] = None,
        t_max: Optional[sc.Variable] = None,
        wav_min: Optional[sc.Variable] = None,
        wav_max: Optional[sc.Variable] = None,
        neutrons: int = 1_000_000,
        sampling: int = 10000,
    ):
        """
        Create a pulse from a pre-defined pulse from a neutron facility.

        Parameters
        ----------
        kind:
            Name of a pre-defined pulse from a neutron facility.
        t_min:
            Start time of the pulse.
        t_max:
            End time of the pulse.
        wav_min:
            Minimum wavelength of the pulse.
        wav_max:
            Maximum wavelength of the pulse.
        neutrons:
            Number of neutrons in the pulse.
        sampling:
            Number of points used to sample the probability distributions.
        """
        pulse = cls(
            t_min=t_min,
            t_max=t_max,
            wav_min=wav_min,
            wav_max=wav_max,
            neutrons=neutrons,
            generate=False,
        )
        pulse.kind = kind
        params = getattr(facilities, pulse.kind)
        if pulse.t_min is None:
            pulse.t_min = params.time.coords['time'].min()
        if pulse.t_max is None:
            pulse.t_max = params.time.coords['time'].max()
        if pulse.wav_min is None:
            pulse.wav_min = params.wavelength.coords['wavelength'].min()
        if pulse.wav_max is None:
            pulse.wav_max = params.wavelength.coords['wavelength'].max()
        pulse.t_min = pulse.t_min.to(unit='s')
        pulse.t_max = pulse.t_max.to(unit='s')
        pulse.wav_min = pulse.wav_min.to(unit='angstrom')
        pulse.wav_max = pulse.wav_max.to(unit='angstrom')

        x_time = np.linspace(pulse.t_min.value, pulse.t_max.value, sampling)
        x_wav = np.linspace(pulse.wav_min.value, pulse.wav_max.value, sampling)
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
            t_min=p_time.coords['time'].min().to(unit='s'),
            t_max=p_time.coords['time'].max().to(unit='s'),
            wav_min=p_wav.coords['wavelength'].min().to(unit='angstrom'),
            wav_max=p_wav.coords['wavelength'].max().to(unit='angstrom'),
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
            x_time = np.linspace(pulse.t_min.value, pulse.t_max.value, sampling)
            x_wav = np.linspace(pulse.wav_min.value, pulse.wav_max.value, sampling)
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
        return self.t_max - self.t_min

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
            t_min=self.t_min,
            t_max=self.t_max,
            wav_min=self.wav_min,
            wav_max=self.wav_max,
        )

    def __repr__(self) -> str:
        return (
            f"Pulse(t_min={self.t_min:c}, t_max={self.t_max:c}, "
            f"wav_min={self.wav_min:c}, wav_max={self.wav_max:c}, "
            f"neutrons={self.neutrons}, kind={self.kind})"
        )


@dataclass(frozen=True)
class ReadonlyPulse:
    birth_times: sc.DataArray
    wavelengths: sc.DataArray
    speeds: sc.DataArray
    kind: Optional[str]
    neutrons: int
    t_min: sc.Variable
    t_max: sc.Variable
    wav_min: sc.Variable
    wav_max: sc.Variable
