import numpy as np
from scipy import fftpack

def multiply(amps1, amps2):
    return amps1*amps2

def power(amps1):
    return amps1*amps1

def subtract(amps1, amps2):
    return amps1 - amps2

def add(amps1, amps2):
    return amps1 + amps2

def derivative(amps1, m=1):
    return fftpack.diff(amps1, order=m)

def integrate(amps1, m=1):
    return fftpack.diff(amps1, order=-m)

def fit(amps1):
    pass

def val(times, amps1):
    return fftpack.irfft(amps1.T, axis=-1)

