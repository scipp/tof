import numpy as np
import matplotlib.pyplot as plt

from . import facilities
from .tools import Plot
from . import units


class Pulse:
    def __init__(
        self,
        tmin: float = None,
        tmax: float = None,
        lmin: float = None,
        lmax: float = None,
        neutrons=1_000_000,
        kind=None,
        p_wav=None,
        p_time=None,
        sampling_resolution=10000,
    ):
        self.kind = kind
        self.neutrons = neutrons

        if self.kind is not None:
            params = getattr(facilities, self.kind)
            self.tmin = params['time'][:, 0].min()
            self.tmax = params['time'][:, 0].max()
            self.lmin = params['wavelength'][:, 0].min()
            self.lmax = params['wavelength'][:, 0].max()

            x_time = np.linspace(self.tmin, self.tmax, sampling_resolution)
            x_wav = np.linspace(self.lmin, self.lmax, sampling_resolution)
            p_time = np.interp(x_time, params['time'][:, 0], params['time'][:, 1])
            p_time /= p_time.sum()
            p_wav = np.interp(
                x_wav, params['wavelength'][:, 0], params['wavelength'][:, 1]
            )
            p_wav /= p_wav.sum()
        else:
            self.tmin = units.us_to_s(tmin)
            self.tmax = units.us_to_s(tmax)
            self.lmin = lmin  # Angstrom
            self.lmax = lmax  # Angstrom

        if p_time is not None:
            if self.kind is None:
                x_time = np.linspace(self.tmin, self.tmax, len(p_time))
            self.birth_times = np.random.choice(x_time, size=self.neutrons, p=p_time)
        else:
            self.birth_times = np.random.uniform(self.tmin, self.tmax, self.neutrons)
        if p_wav is not None:
            if self.kind is None:
                x_wav = np.linspace(self.lmin, self.lmax, len(p_wav))
            self.wavelengths = np.random.choice(x_wav, size=self.neutrons, p=p_wav)
        else:
            self.wavelengths = np.random.uniform(self.lmin, self.lmax, self.neutrons)

        self.speeds = units.wavelength_to_speed(self.wavelengths)
        self.energies = units.speed_to_mev(self.speeds)

    def __repr__(self):
        return (
            f"Pulse(tmin={self.tmin}, tmax={self.tmax}, lmin={self.lmin}, "
            f"lmax={self.lmax}, neutrons={self.neutrons}, kind={self.kind})"
        )

    @property
    def duration(self):
        return self.tmax - self.tmin

    def plot(self, bins=300):
        fig, ax = plt.subplots(1, 2)
        for i, (data, label) in enumerate(
            zip([self.birth_times, self.wavelengths], ["Time", "Wavelength"])
        ):
            h, edges = np.histogram(data, bins=bins)
            x = np.concatenate([edges, edges[-1:]])
            y = np.concatenate([[0], h, [0]])
            ax[i].step(x, y)
            ax[i].fill_between(x, 0, y, step="pre", alpha=0.5)
            ax[i].set_xlabel(label)
            ax[i].set_ylabel("Counts")
        size = fig.get_size_inches()
        fig.set_size_inches(size[0] * 2, size[1])
        return Plot(fig=fig, ax=ax)
