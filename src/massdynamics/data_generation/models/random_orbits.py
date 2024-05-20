import numpy as np
from massdynamics.basis_functions import basis
import massdynamics.window_functions as window_functions
from massdynamics.data_generation import (
    data_generation,
    compute_waveform,
    data_processing
)

def generate_random_coefficients(
    order: int, 
    n_dimensions: int = 1, 
    basis_type:str = "chebyshev",
    fourier_weight: float = 0.0) -> tuple:
    """_summary_

    Args:
        order (int): _description_

    Returns:
        tuple: (order, dimensions)
    """

    if basis_type == "chebyshev":
        coefficients = np.zeros((order,n_dimensions))
        for i in range(order):
            coefficients[i] = 2 * np.random.rand(n_dimensions) - 1
    elif basis_type == "fourier":
        if order % 2 != 0:
            raise Exception(f"Please specify an order that is divisible by 2 for fourier basis, currently {order}")
        halforder = int(order/2) + 1
        coefficients = np.zeros((halforder,n_dimensions), dtype=basis[basis_type]["dtype"])
        for i in range(halforder):
            coefficients[i] = np.exp(-fourier_weight*i) * (2*np.random.rand(n_dimensions)-1 + 1j*(2 * np.random.rand(n_dimensions) - 1))

    return coefficients




def generate_masses(n_masses: int, data_type="random-uniform", prior_args={}) -> np.array:
    """generate masses 

    Args:
        n_masses (int): _description_

    Returns:
        np.array: _description_
    """
    
    masses = np.random.uniform(prior_args["masses_min"], prior_args["masses_max"], n_masses)
    if data_type.split("-")[1] == "equalmass":
        masses[1] = masses[0]

    if data_type.split("-")[1] == "2masstriangle":
        m1 = masses[0]
        m2 = masses[1]
        if m2 > m1:
            masses = np.array([m2, m1])
    masses = masses/np.sum(masses)

    return masses

def generate_data(
    n_data: int, 
    basis_order: int, 
    n_masses:int, 
    sample_rate: int, 
    n_dimensions: int = 1, 
    detectors=["H1"], 
    window="none", 
    window_acceleration=True, 
    basis_type="chebyshev",
    fourier_weight=0.0,
    data_type = "random-uniform",
    prior_args = {}) -> np.array:
    """_summary_

    Args:
        n_data (int): number of data samples to generate
        n_order (int): order of polynomials 
        n_masses (int): number of masses in system
        sample_rate (int): sample rate of data

    Returns:
        np.array: _description_
    """

    if basis_type == "fourier":
        dtype = complex
    else:
        dtype = np.float64

    prior_args.setdefault("masses_min", 0)
    prior_args.setdefault("masses_max", 1)
    prior_args.setdefault("fourier_weight", 0.0)
    ntimeseries = [0, 1, 3, 6, 10]

    strain_timeseries = np.zeros((n_data, len(detectors), sample_rate))

    times = np.arange(0,1,1./sample_rate)

    random_coeffs = generate_random_coefficients(
        basis_order, 
        n_dimensions,
        basis_type=basis_type,
        fourier_weight=fourier_weight)

    if window_acceleration not in [False, None, "none"]:
        coeffs, window_coeffs = window_functions.perform_window(times, random_coeffs, window_acceleration, basis_type=basis_type, order=basis_order)
    else:
        coeffs = random_coeffs

    if window_acceleration:  
        win_basis_order = np.shape(coeffs)[0]
        acc_basis_order = np.shape(coeffs)[0]
    else:
        win_basis_order = np.shape(coeffs)[0]
        acc_basis_order = basis_order

        if basis_type == "fourier":
            # plus 2 as plus 1 in the real and imaginary in coeff gen
            # this is so can get back to order is 1/2 n_samples
            acc_basis_order += 2

    if basis_type == "fourier":
        output_coeffs_mass = np.zeros((n_data, acc_basis_order*n_masses*n_dimensions + n_masses))
    else:
        output_coeffs_mass = np.zeros((n_data, acc_basis_order*n_masses*n_dimensions + n_masses))

    all_time_dynamics = np.zeros((n_data, n_masses, n_dimensions, len(times)))
    all_basis_dynamics = np.zeros((n_data, n_masses, n_dimensions, win_basis_order), dtype=dtype)
    all_masses = np.zeros((n_data, n_masses))

    for data_index in range(n_data):

        masses = generate_masses(n_masses, data_type=data_type, prior_args=prior_args)
        all_masses[data_index] = masses

        #all_dynamics = np.zeros((n_masses, n_dimensions, win_basis_order), dtype=dtype)
        #output_coeffs_mass[data_index, -n_masses:] = masses
        #temp_output_coeffs = np.zeros((n_masses, n_dimensions, acc_basis_order))
        for mass_index in range(n_masses):

            random_coeffs = generate_random_coefficients(
                basis_order, 
                n_dimensions,
                basis_type = basis_type,
                fourier_weight=fourier_weight)

            if window != "none":
                coeffs  = window_functions.window_coeffs(times, random_coeffs, window_coeffs, basis_type=basis_type)
            else:
                coeffs = random_coeffs
                
            all_basis_dynamics[data_index, mass_index] = coeffs.T 
        
        
    """
            # if windowing applied create coeffs which are windowed else just use the random coeffs
            if window != "none":
                coeffs = window_coeffs(times, random_coeffs, win_coeffs, basis_type=basis_type)
            else:
                coeffs = random_coeffs

            all_basis_dynamics[data_index, mass_index] = coeffs.T 

        strain_timeseries[data_index], energy = compute_waveform.get_waveform(
            times, 
            masses, 
            all_basis_dynamics[data_index], 
            detectors, 
            basis_type=basis_type,
            compute_energy=False)

        all_time_dynamics[data_index] = compute_waveform.get_time_dynamics(
            all_basis_dynamics[data_index], 
            times, 
            basis_type=basis_type)


    output_coeffs_mass = data_processing.positions_masses_to_samples(
        all_basis_dynamics,
        all_masses,
        basis_type = basis_type
        )
    """


    return times, None, all_masses, all_basis_dynamics
    #return times, output_coeffs_mass, strain_timeseries, acc_basis_order, all_time_dynamics, all_basis_dynamics
