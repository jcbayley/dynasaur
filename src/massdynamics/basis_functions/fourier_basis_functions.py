import numpy as np
from scipy import fftpack, signal

def multiply(amps1, amps2):
    if type(amps2) in [np.array, np.ndarray, list]:
        ift = np.fft.irfft(amps1)*np.fft.irfft(amps2)
        window = 1.0#signal.windows.tukey(np.shape(ift)[-1], alpha=1.0)
        return np.fft.rfft(ift*window)
        #return np.convolve(amps1, amps2, mode="full")
    else:
        return amps1 * amps2

def power(amps1, p=2):
    ift = np.fft.irfft(amps1)*np.fft.irfft(amps1)
    window = 1.0#signal.windows.tukey(np.shape(ift)[-1], alpha=1.0)
    return np.fft.rfft(ift*window)
    #return np.convolve(amps1, amps1, mode="full")

def subtract(amps1, amps2):
    return amps1 - amps2

def add(amps1, amps2):
    return amps1 + amps2

def derivative(amps1, m=1, duration=2):
    # needs to be fixed to take in different durations
    freqs = np.arange(len(amps1))/duration
    output = amps1 * (freqs * 2 * np.pi * 1j)**m
    return output

def integrate(amps1, m=1, duration=2):

    # needs to be fixed to take in different durations
    freqs = np.arange(len(amps1))/duration
    output = amps1 * (freqs * 2 * np.pi * 1j)**(-m)
    return output

def fit(times, amps1, order):
    """ find inverse of function (inverse fft)

    Args:
        times (_type_): (Ntimes)
        amps1 (_type_): (Ndets, Ntimes)
        order (_type_): order of basis to take

    Returns:
        _type_: _description_
    """
    window = 1+1j#signal.windows.tukey(np.shape(amps1)[-1], alpha=0.5)
    fftout = np.fft.rfft(amps1*window, axis=-1)[...,:order]#*int(len(times)/2 + 1)/order
    return fftout

def val(times, amps1):
    # pad the frequency series with zeros to get same timeseries back
    # amps1 shape should have the frequency dimension as 0
    
    # compute ratio for renormalisiation later
    shape_ratio = int(len(times)/2 + 1) / np.shape(amps1)[0]
    """
    if int(len(times)/2 + 1) > np.shape(amps1)[0]:
        tempshape = np.array(np.shape(amps1))
        # add on zeros so half langth of ts (ts will then be correct size)
        tempshape[0] = int(len(times)/2 + 1) - np.shape(amps1)[0]
        zerosappend = np.zeros(tuple(tempshape)).astype(complex)
        amps1 = np.concatenate([amps1, zerosappend], axis=0)

    fftout = np.fft.irfft(amps1, axis=0) * shape_ratio
    """
    fftout = np.fft.irfft(amps1, n=len(times), axis=-1)#*1./shape_ratio
    # switch back to having the time dimension last
    # this is so its consistent with the np polynomial val function
    #
    return fftout

