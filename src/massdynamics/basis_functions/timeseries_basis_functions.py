import numpy as np
from scipy import fftpack, signal

def multiply(amps1, amps2):
        return amps1 * amps2

def power(amps1, p=2):
    return amps1*amps1
    #return np.convolve(amps1, amps1, mode="full")

def subtract(amps1, amps2):
    return amps1 - amps2

def add(amps1, amps2):
    return amps1 + amps2

def derivative(amps1, m=1, duration=2):
    # needs to be fixed to take in different durations
    return np.gradient(amps1)

def integrate(amps1, m=1, duration=2):

    # needs to be fixed to take in different durations
    #dt = t[1] - t[0]
    dt = 1
    if m == 1:
        amps_int = np.cumsum(amps1) * dt + amps1[0]
    elif m == 2:
        amps_int = np.cumsum(amps1) * dt + amps1[0]
        amps_int = np.cumsum(amps_int) * dt + amps_int[0]
    return amps_int

def fit(times, amps1, order):
    """ find inverse of function (inverse fft)

    Args:
        times (_type_): (Ntimes)
        amps1 (_type_): (Ndets, Ntimes)
        order (_type_): order of basis to take

    Returns:
        _type_: _description_
    """
    return amps1

def val(times, amps1):
    """find val, in this case it ios jsut the timesereis

    Args:
        times (_type_): _description_
        amps1 (_type_): _description_

    Returns:
        _type_: _description_
    """
    return amps1

