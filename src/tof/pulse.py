import numpy as np

from . import units


class Pulse:
    def __init__(self, stop: float = 1.0, start: float = 0.0, kind="ess"):
        self.start = units.us_to_s(start)
        self.stop = units.us_to_s(stop)
        self.kind = kind

        if self.kind == "ess":
            self.stop = self.start + units.us_to_s(2860.0)

        self.birth_times = None
        self.wavelengths = None
        self.speeds = None

        self.wavelength_min = 1.0  # Angstrom
        self.wavelength_max = 10.0  # Angstrom

    def __repr__(self):
        return f"Pulse(start={self.start}, stop={self.stop}, kind={self.kind})"

    @property
    def duration(self):
        return self.stop - self.start

    def make_neutrons(self, n):
        self.birth_times = np.random.uniform(self.start, self.stop, n)
        self.wavelengths = np.random.uniform(
            self.wavelength_min, self.wavelength_max, n
        )
        self.speeds = units.wavelength_to_speed(self.wavelengths)
