import numpy as np
from scipy import fftpack

def multiply(amps1, amps2):
    if type(amps2) in [np.array, np.ndarray, list]:
        return np.fft.rfft(np.fft.irfft(amps1)*np.fft.irfft(amps2))
    else:
        return amps1 * amps2

def power(amps1):
    return np.fft.rfft(np.fft.irfft(amps1)*np.fft.irfft(amps1))

def subtract(amps1, amps2):
    return amps1 - amps2

def add(amps1, amps2):
    return amps1 + amps2

def derivative(amps1, m=1, freqs=np.arange(33)*2./2):
    output = amps1 * (freqs * 2 * np.pi * 1j)**m
    return output

def integrate(amps1, m=1, freqs=np.arange(33)*2./2):
    output = amps1 * (freqs * 2 * np.pi * 1j)**(-m)
    return output

def fit(amps1):
    return np.fft.rfft(amps1)

def val(times, amps1):
    return np.fft.irfft(amps1.T, axis=-1)
