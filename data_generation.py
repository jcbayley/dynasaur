import numpy as np

def generate_random_coefficients(order: int, n_dimensions: int = 1) -> tuple:
    """_summary_

    Args:
        order (int): _description_

    Returns:
        tuple: _description_
    """

    coefficients = np.zeros((order,n_dimensions))

    for i in range(order):
        coefficients[i] = 2 * np.random.rand(n_dimensions) - 1

    return coefficients

def generate_strain_coefficients(coeffs: np.array) -> np.array:
    """

    Args:
        coeffs (np.array): _description_

    Returns:
        np.array: _description_
    """

    squares_chebyshev = np.polynomial.chebyshev.chebpow(coeffs, 2)
    diff_chebyshev = np.polynomial.chebyshev.chebder(squares_chebyshev, m=2)

    return diff_chebyshev

def generate_2d_derivative(coeffs: np.array, times: np.array) -> np.array:
    """

    Args:
        coeffs (np.array): _description_

    Returns:
        np.array: _description_
    """

    # this is the second mass moment (quadrupole moment)
    # making trace free to get quadrupole moment
    # subtract 0.5*I_ii is the same as multiplying the diagonal by 0.5
    co_xx = np.polynomial.chebyshev.chebpow(coeffs[:,0], 2) * 0.5
    co_yy = np.polynomial.chebyshev.chebpow(coeffs[:,1], 2) * 0.5
    co_xy = np.polynomial.chebyshev.chebmul(coeffs[:,0], coeffs[:,1])
    co_yx = np.polynomial.chebyshev.chebmul(coeffs[:,1], coeffs[:,0])

    # compute derivatives of quadrupole
    h_xx = np.polynomial.chebyshev.chebder(co_xx, m=2)
    h_yy = np.polynomial.chebyshev.chebder(co_yy, m=2)
    h_xy = np.polynomial.chebyshev.chebder(co_xy, m=2)
    h_yx = np.polynomial.chebyshev.chebder(co_yx, m=2)

    # compute x and y as timeseries
    x = np.polynomial.chebyshev.chebval(times, coeffs[:,0])
    y = np.polynomial.chebyshev.chebval(times, coeffs[:,1])

    # projection tensor
    P = np.array([
        [1 - x*x, -x*y],
        [-x*y, 1 - y*y]
    ])

    # get quadrupole moment as time series
    Iprime_coeffs = np.array([[h_xx, h_xy],[h_yx, h_yy]])
    Iprime = np.zeros((2,2,len(times)))
    for i in range(2):
        for j in range(2):
            Iprime[i,j] = np.polynomial.chebyshev.chebval(times, Iprime_coeffs[i,j])

    # compute the TT gauge strain tensor as projection tensor
    # see https://arxiv.org/pdf/gr-qc/0501041.pdf
    h_TT = np.zeros((2,2,len(times)))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    h_TT[i,j] += Iprime[k,l] * (P[i,k]*P[j,l] - 0.5*P[k,l]*P[i,j])

    return h_TT

def generate_masses(n_masses: int) -> np.array:
    """_summary_

    Args:
        n_masses (int): _description_

    Returns:
        np.array: _description_
    """

    masses = np.random.rand(n_masses)
    masses = masses/np.sum(masses)

    return masses

def generate_data(n_data: int, chebyshev_order: int, n_masses:int, sample_rate: int, n_dimensions: int = 1) -> np.array:
    """_summary_

    Args:
        n_data (int): _description_
        n_order (int): _description_
        n_masses (int): 
        sample_rate (int): _description_

    Returns:
        np.array: _description_
    """

    ntimeseries = [0, 1, 3, 6, 10]

    strain_timeseries = np.zeros((n_data, sample_rate))
    flattened_coeffs_mass = np.zeros((n_data, chebyshev_order*n_masses*n_dimensions + n_masses))

    times = np.arange(-1,1,2/sample_rate)

    for data_index in range(n_data):

        masses = generate_masses(n_masses)

        all_dynamics = np.zeros(chebyshev_order*n_dimensions)
        flattened_coeffs_mass[data_index, -n_masses:] = masses
        for mass_index in range(n_masses):

            coeffs = generate_random_coefficients(chebyshev_order, n_dimensions)
            flat_coeffs = np.ravel(coeffs)

            flattened_coeffs_mass[data_index, chebyshev_order*mass_index*n_dimensions:chebyshev_order*n_dimensions*(mass_index+1)] = flat_coeffs
            all_dynamics += masses[mass_index]*flat_coeffs

        if n_dimensions == 1:
            strain_coeffs = generate_strain_coefficients(all_dynamics)
            strain_timeseries[data_index] = np.polynomial.chebyshev.chebval(times, strain_coeffs)
        elif n_dimensions == 2:
            temp_strain_timeseries = generate_2d_derivative(all_dynamics.reshape(chebyshev_order, n_dimensions), times)
            hplus = temp_strain_timeseries[0,0]
            hcross = temp_strain_timeseries[0,1]
            strain_timeseries[data_index] = hplus + hcross
    return times, flattened_coeffs_mass, strain_timeseries

