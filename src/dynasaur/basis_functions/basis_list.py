import numpy as np
from dynasaur.basis_functions import (
    fourier_basis_functions,
    chebyshev_basis_functions,
    timeseries_basis_functions)

basis = {
    "chebyshev":{
        "multiply": chebyshev_basis_functions.multiply,
        "power": chebyshev_basis_functions.power,
        "subtract": chebyshev_basis_functions.subtract,
        "add": chebyshev_basis_functions.add,
        "derivative": chebyshev_basis_functions.derivative,
        "integrate": chebyshev_basis_functions.integrate,
        "fit": chebyshev_basis_functions.fit,
        "val": chebyshev_basis_functions.val,
        "dtype": np.float64
    },
    "fourier":{
        "multiply": fourier_basis_functions.multiply,
        "power": fourier_basis_functions.power,
        "subtract": fourier_basis_functions.subtract,
        "add": fourier_basis_functions.add,
        "derivative": fourier_basis_functions.derivative,
        "integrate": fourier_basis_functions.integrate,
        "fit": fourier_basis_functions.fit,
        "val": fourier_basis_functions.val,
        "dtype": complex
    },
    "timeseries":{
        "multiply": timeseries_basis_functions.multiply,
        "power": timeseries_basis_functions.power,
        "subtract": timeseries_basis_functions.subtract,
        "add": timeseries_basis_functions.add,
        "derivative": timeseries_basis_functions.derivative,
        "integrate": timeseries_basis_functions.integrate,
        "fit": timeseries_basis_functions.fit,
        "val": timeseries_basis_functions.val,
        "dtype": complex
    }
}