import numpy as np

def generate_random_coefficients(order: int) -> tuple:
    """_summary_

    Args:
        order (int): _description_

    Returns:
        tuple: _description_
    """

    x_coefficients = np.zeros(order)
    #y_coefficients = np.zeros(order)
    #z_coefficients = np.zeros(order)

    for i in range(order):
        x_coefficients = 2 * np.random.rand() - 1

    return x_coefficients

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

def generate_data(n_data: int, chebyshev_order: int, n_masses:int, sample_rate: int ) -> np.array:
    """_summary_

    Args:
        n_data (int): _description_
        n_order (int): _description_
        n_masses (int): 
        sample_rate (int): _description_

    Returns:
        np.array: _description_
    """

    cheby_weights = np.zeros((n_data, n_masses))
    strain_timeseries = np.zeros((n_data, sample_rate))
    flattened_coeffs_mass = np.zeros((n_data, chebyshev_order*n_masses + n_masses))

    times = np.arange(-1,1,2/sample_rate)

    for data_index in range(n_data):

        masses = generate_masses(n_masses)

        all_dynamics = np.zeros(chebyshev_order)
        flattened_coeffs_mass[data_index, -n_masses:] = masses
        for mass_index in range(n_masses):

            coeffs = generate_random_coefficients(chebyshev_order)

            flattened_coeffs_mass[data_index, chebyshev_order*mass_index:chebyshev_order*(mass_index+1)] = coeffs
            all_dynamics += masses[mass_index]*coeffs

        strain_coeffs = generate_strain_coefficients(all_dynamics)

        strain_timeseries[data_index] = np.polynomial.chebyshev.chebval(times, strain_coeffs)

    return times, flattened_coeffs_mass, strain_timeseries

