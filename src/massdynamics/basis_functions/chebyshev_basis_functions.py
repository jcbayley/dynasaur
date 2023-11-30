import numpy as np
from scipy import fftpack, signal

def multiply(amps1, amps2):
    return np.polynomial.chebyshev.chebmul(amps1, amps2)

def power(amps1):
    return np.polynomial.chebyshev.chebpow(amps1)

def subtract(amps1, amps2):
    return np.polynomial.chebyshev.chebsub(amps1, amps2)

def add(amps1, amps2):
    return np.polynomial.chebyshev.chebadd(amps1, amps2)

def derivative(amps1, m=1, duration=2):
    # needs to be fixed to take in different durations
    return np.polynomial.chebyshev.chebder(amps1, m=m)

def integrate(amps1, m=1, duration=2):

    return np.polynomial.chebyshev.chebint(amps1, m=m)

def fit(times, amps1, order):
    """ find inverse of function (inverse fft)

    Args:
        times (_type_): (Ntimes)
        amps1 (_type_): (Ndets, Ntimes)
        order (_type_): order of basis to take

    Returns:
        _type_: (Ndets, Ncoeffs)
    """
    amps_fit = np.transpose(amps1, (1,0))
    fit = np.transpose(np.polynomial.chebyshev.chebfit(times, amps_fit, order), (1,0))
    return fit

def val(times, amps1):
    """_summary_

    Args:
        times (_type_): Ntimes
        amps1 (_type_): (Ndims, Ndims, Ncoeffs)

    Returns:
        _type_: (Ndims, Ndims, Ntimes)
    """
    # pad the frequency series with zeros to get same timeseries back
    # amps1 shape should have the frequency dimension as 0
    
    # transpose as applied over first dimension but returns over last
    fftout = np.transpose(amps1, (2, 0, 1))
    return np.polynomial.chebyshev.chebval(times, fftout)

