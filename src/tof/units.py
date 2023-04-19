import numpy as np

mass = 1.674927471e-27  # Neutron mass in kg
alpha = 2.5278e-4  # Neutron mass over Planck constant
mev = 1.602176634e-22  # meV to Joule
# angstrom = 1.0e-10  # Angstrom to meter


def deg_to_rad(x):
    return np.radians(x)


def rad_to_deg(x):
    return np.degrees(x)


def us_to_s(x):
    return x * 1.0e-6


def s_to_us(x):
    return x * 1.0e6


def speed_to_wavelength(x):
    return 1.0 / (alpha * x)


def wavelength_to_speed(x):
    return 1.0 / (alpha * x)


def speed_to_mev(x):
    return mass * x * x / mev


def mev_to_speed(x):
    return np.sqrt(mev * x / mass)
