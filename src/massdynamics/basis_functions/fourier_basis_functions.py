import numpy as np
from scipy import fftpack, signal

def multiply(amps1, amps2):
    if type(amps2) in [np.array, np.ndarray, list]:
        ift = np.fft.irfft(amps1)*np.fft.irfft(amps2)
        window = 1#signal.windows.tukey(np.shape(ift)[-1], alpha=0.5)
        return np.fft.rfft(ift*window)
    else:
        return amps1 * amps2

def power(amps1):
    ift = np.fft.irfft(amps1)*np.fft.irfft(amps1)
    window = 1#signal.windows.tukey(np.shape(ift)[-1], alpha=0.5)
    return np.fft.rfft(ift*window)

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
    window = 1#signal.windows.tukey(np.shape(amps1)[-1], alpha=0.5)
    return np.fft.rfft(amps1*window)

def val(times, amps1):
    return np.fft.irfft(amps1.T, axis=-1)

