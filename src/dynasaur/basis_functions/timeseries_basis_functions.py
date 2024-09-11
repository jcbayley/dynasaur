import numpy as np
from scipy import fftpack, signal
from scipy.integrate import cumtrapz

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
    output = amps1
    for i in range(m):
        output = np.gradient(output, axis=-1)
    return output

def int1(t1, t0):
    rt0 = cumtrapz(t1, axis=-1) + np.tile(t0[...,0:1], (np.shape(t0)[-1]-1))
    rt0 = np.insert(rt0, 0, t0[...,0], axis=-1)
    return rt0

def integrate(amps1, amps0=None, amps05=None,  m=1, duration=2):

    # needs to be fixed to take in different durations
    #dt = t[1] - t[0]
    dt = 1
    if m == 1:
        if amps0 is not None:
            amps_int = int1(amps1, amps0)
        else:
            amps_int = cumtrapz(amps1, axis=-1)
    elif m == 2:
        if amps0 is not None:
            amps_int = int1(amps1, amps0)
        else:
            amps_int = cumtrapz(amps1, axis=-1)
        if amps05 is not None:
            amps_int = int1(amps_int, amps05)
        else:
            amps_int = cumtrapz(amps_int, axis=-1)
            
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

