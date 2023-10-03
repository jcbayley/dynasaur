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
    """ takes in coefficients for polynomial in two dimensions
        -> computes quadrupole tensor
        -> computes gravitational pertubation tensor
        -> projects into TT gauge to get hplus and hcross
        -> returns hplus + hcross
        
        will eventually include detector tensors (antenna pattern functions etc)

    Args:
        coeffs (np.array): 2d array of coefficients for x and y

    Returns:
        np.array: h(t) time series of GW
    """

    # this is the second mass moment (quadrupole moment)
    # making trace free to get quadrupole moment
    # subtract 0.5*I_ii is the same as multiplying the diagonal by 0.5
    co_xx = np.polynomial.chebyshev.chebpow(coeffs[:,0], 2) 
    co_yy = np.polynomial.chebyshev.chebpow(coeffs[:,1], 2) 
    
    # subtract the trace
    trace = np.polynomial.chebyshev.chebadd(co_xx, co_yy)
    factor = np.polynomial.chebyshev.chebmul(trace, 0.5)
    co_xx = np.polynomial.chebyshev.chebsub(co_xx, factor)
    co_yy = np.polynomial.chebyshev.chebsub(co_yy, factor)
    
    # compute off diagonals
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

    # have to do this in the time series as cant find sqrt of chebyshev as coefficients
    r = np.sqrt(x**2 + y**2)
    normx = x/r
    normy = y/r

    # projection tensor
    P = np.array([
        [1 - normx*normx, -normx*normy],
        [-normx*normy, 1 - normy*normy]
    ])

    """
    # this step of taking square root not working
    r = np.polynomial.chebyshev.chebpow(np.polynomial.chebyshev.chebadd(co_xx, co_xy), 0.5)
    normx = np.polynomial.chebyshev.chebdiv(coeffs[:,0], r)
    normy = np.polynomial.chebyshev.chebdiv(coeffs[:,1], r)

    P_xx = np.polynomial.chebyshev.chebadd(np.polynomial.chebyshev.chebmul(np.polynomial.chebyshev.chebpow(normx, 2), -1), 1)
    P_yy = np.polynomial.chebyshev.chebadd(np.polynomial.chebyshev.chebmul(np.polynomial.chebyshev.chebpow(normy, 2), -1), 1)
    P_xy = np.polynomial.chebyshev.chebmul(np.polynomial.chebyshev.chebmul(normx, normy), -1)
    P_yx = np.polynomial.chebyshev.chebmul(np.polynomial.chebyshev.chebmul(normy, normx), -1)


    P_coeffs = np.array([
        [P_xx, P_xy],
        [P_yx, P_yy]
    ])
    

    I_coeffs = np.array([[co_xx, co_xy],[co_yx, co_yy]])
    ITT_coeffs = np.zeros((2,2, 43))
    for i in range(2):
        for j in range(2):
            fact1 = np.zeros(43)
            fact2 = np.zeros(43)
            for k in range(2):
                for l in range(2):
                    t_fact1 = np.polynomial.chebyshev.chebmul(P_coeffs[i,k] , I_coeffs[k,l])
                    t_fact1 = np.polynomial.chebyshev.chebmul(t_fact1, P_coeffs[l,j])
                    fact1 = np.polynomial.chebyshev.chebadd(fact1, t_fact1)

                    t_fact2 = np.polynomial.chebyshev.chebmul(P_coeffs[k,l], I_coeffs[k,l])
                    fact2 = np.polynomial.chebyshev.chebadd(fact2, t_fact2)

            fact2 = np.polynomial.chebyshev.chebmul(P_coeffs[i,j] , fact2)
            fact2 = np.polynomial.chebyshev.chebmul(fact2 , -0.5)
            total = np.polynomial.chebyshev.chebadd(fact1, fact2)
            ITT_coeffs[i,j] = np.polynomial.chebyshev.chebadd(fact1, fact2)


    h_TT_coeffs = np.zeros((2,2,41))
    h_TT = np.zeros((2,2,len(times)))
    for i in range(2):
        for j in range(2):
            h_TT_coeffs[i,j] = np.polynomial.chebyshev.chebder(ITT_coeffs[i,j], m=2)
            h_TT[i,j] = np.polynomial.chebyshev.chebval(times, h_TT_coeffs[i,j])
    """
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
            fact1 = np.zeros(len(times))
            fact2 = np.zeros(len(times))
            for k in range(2):
                for l in range(2):
                    fact1 += Iprime[k,l] * P[k,l]
                    fact2 += Iprime[k,l] * P[i,k] * P[j,l]
                    
            h_TT[i,j] = fact2 - 0.5*fact1
    
    #print("hxx", h_TT[0,0])
    #print("hxy", h_TT[1,1])

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

if __name__ == "__main__":

    n_masses = 2
    chebyshev_order = 8
    n_dimensions = 2
    sample_rate = 32
    times = np.arange(-1,1,2/sample_rate)
    masses = generate_masses(n_masses)

    all_dynamics = np.zeros(chebyshev_order*n_dimensions)
    flattened_coeffs_mass = np.zeros((chebyshev_order*n_masses*n_dimensions + n_masses))

    flattened_coeffs_mass[-n_masses:] = masses
    for mass_index in range(n_masses):

        coeffs = generate_random_coefficients(chebyshev_order, n_dimensions)
        flat_coeffs = np.ravel(coeffs)

        flattened_coeffs_mass[chebyshev_order*mass_index*n_dimensions:chebyshev_order*n_dimensions*(mass_index+1)] = flat_coeffs
        all_dynamics += masses[mass_index]*flat_coeffs

    temp_strain_timeseries = generate_2d_derivative(all_dynamics.reshape(chebyshev_order, n_dimensions), times)
    print(temp_strain_timeseries[0,0] + temp_strain_timeseries[1,1])
    print(temp_strain_timeseries[0,1] - temp_strain_timeseries[1,0])
    hplus = temp_strain_timeseries[0,0]
    hcross = temp_strain_timeseries[0,1]
    