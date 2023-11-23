import numpy as np
from massdynamics.basis_functions import fourier_basis_functions

basis = {
    "chebyshev":{
        "multiply": np.polynomial.chebyshev.chebmul,
        "power": np.polynomial.chebyshev.chebpow,
        "subtract": np.polynomial.chebyshev.chebsub,
        "add": np.polynomial.chebyshev.chebadd,
        "derivative": np.polynomial.chebyshev.chebder,
        "integrate": np.polynomial.chebyshev.chebint,
        "fit": np.polynomial.chebyshev.chebfit,
        "val": np.polynomial.chebyshev.chebval,
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
    }
}