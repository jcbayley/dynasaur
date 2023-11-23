import numpy as np
import lal
import lalpulsar 
import matplotlib.pyplot as plt
import scipy.signal as signal

polynomial_dict = {
    "chebyshev":{
        "multiply": np.polynomial.chebyshev.chebmul,
        "power": np.chebyshev.polynomial.chebpow,
        "subtract": np.polynomial.chebyshev.chebsub,
        "add": np.polynomial.chebyshev.chebadd,
        "derivative": np.polynomial.chebyshev.chebpow,
        "integrate": np.polynomial.chebyshev.int,
        "fit": np.polynomial.chebyshev.fit,
        "val": np.polynomial.chebyshev.chebval,
    }
}

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

def generate_strain_coefficients(coeffs: np.array, poly_type="chebyshev") -> np.array:
    """

    Args:
        coeffs (np.array): _description_

    Returns:
        np.array: _description_
    """

    squares_chebyshev = polynomial_dict[poly_type]["power"](coeffs, 2)
    diff_chebyshev = polynomial_dict[poly_type]["derivative"](squares_chebyshev, m=2)

    return diff_chebyshev

def fit_cheby_to_hann(times, order=6, poly_type="chebyshev"):
    hwin = np.hanning(len(times))
    hann_cheb = polynomial_dict[poly_type]["fit"](times, hwin, order)
    return hann_cheb

def fit_cheby_to_tukey(times, alpha=0.5, order=6, polytype="chebyshev"):
    hwin = signal.windows.tukey(len(times), alpha=alpha)
    tuk_cheb = polynomial_dict[poly_type]["fit"](times, hwin, order)
    return tuk_cheb


def chebint2(times, coeffs, poly_type="chebyshev"):
    """compute the second integral correcting for offsets in the integrated values

    Args:
        times (_type_): _description_
        coeffs (_type_): _description_

    Returns:
        _type_: _description_
    """
    win_co_vel = polynomial_dict[poly_type]["integrate"](coeffs, m=1)
    # compute the values and subtract the mean from the first coefficient
    # this is so that there is not a velocity offset
    win_vel = polynomial_dict[poly_type]["val"](times, win_co_vel)
    win_co_vel[0] -= np.mean(win_vel)

    # now find the position
    win_co_pos = polynomial_dict[poly_type]["integrate"](win_co_vel, m=1)
    # compute the values and subtract the mean from the first coefficient
    # this is so that there is not a position offset
    win_pos = polynomial_dict[poly_type]["val"](times, win_co_pos)
    win_co_pos[0] -= np.mean(win_pos)

    return win_co_pos

def window_coeffs(times, coeffs, window_coeffs, poly_type="chebyshev"):
    #hann_coeffs = np.array([ 3.47821791e-01,  1.52306260e-16, -4.85560481e-01, -5.11827799e-17, 1.51255010e-01,  2.65316279e-17, -1.48207898e-02])

    # find the acceleration components for each dimension
    co_x_acc = polynomial_dict[poly_type]["derivative"](coeffs[:,0], m=2)
    co_y_acc = polynomial_dict[poly_type]["derivative"](coeffs[:,1], m=2)
    co_z_acc = polynomial_dict[poly_type]["derivative"](coeffs[:,2], m=2)

    # window each dimension in acceleration according to hann window
    win_co_x_acc = polynomial_dict[poly_type]["multiply"](co_x_acc, window_coeffs)
    win_co_y_acc = polynomial_dict[poly_type]["multiply"](co_y_acc, window_coeffs)
    win_co_z_acc = polynomial_dict[poly_type]["multiply"](co_z_acc, window_coeffs)

    # fix bug when object not moving in z axes (repeat 0 for n coeffs)
    if len(win_co_z_acc) == 1:
        win_co_z_acc = np.repeat(win_co_z_acc[0], len(win_co_y_acc))
    # integrate the windowed acceleration twice to get position back
    #win_co_x = polynomial_dict[poly_type]["integrate"](win_co_x_acc, m=2)
    #win_co_y = polynomial_dict[poly_type]["integrate"](win_co_y_acc, m=2)
    #win_co_z = polynomial_dict[poly_type]["integrate"](win_co_z_acc, m=2)

    win_co_x = chebint2(times, win_co_x_acc)
    win_co_y = chebint2(times, win_co_y_acc)
    win_co_z = chebint2(times, win_co_z_acc)

    
    coarr = np.array([win_co_x, win_co_y, win_co_z]).T
    return coarr

def perform_window(times, coeffs, window, order=6, poly_type="chebyshev"):
    """_summary_

    Args:
        times (_type_): _description_
        coeffs (_type_): _description_
        window (_type_): _description_
    """
    if window is not None or window != False:
        if window == "tukey":
            win_coeffs = fit_cheby_to_tukey(times, alpha=0.5, order=order, poly_type=poly_type)
        elif window == "hann":
            win_coeffs = fit_cheby_to_hann(times, order=order, poly_type=poly_type)
        else:
            raise Exception(f"Window {window} does not Exist")

        coeffs = window_coeffs(times, coeffs, win_coeffs)

    else:
        win_coeffs = None

    return coeffs, win_coeffs

def subtract_trace(coeffs, poly_type="chebyshev"):
    """subtract the trace from a 3d tensor

    Args:
        coeffs (_type_): _description_

    Returns:
        _type_: _description_
    """
    # sum diagonal to compute trace
    n_dimensions, n_dimensions, n_coeffs = np.shape(coeffs)
    trace = polynomial_dict[poly_type]["add"](
        polynomial_dict[poly_type]["add"](
            coeffs[0,0], 
            coeffs[1,1]), 
        coeffs[2,2])
    # divide by three the subtract from diagonals
    factor = polynomial_dict[poly_type]["multiply"](trace, 1./3)
    for i in range(n_dimensions):
        coeffs[i,i] = polynomial_dict[poly_type]["subtract"](coeffs[i,i], factor)

    return coeffs

def compute_second_mass_moment(masses, coeffs, remove_trace = False, poly_type="chebyshev"):
    """Performs integral over density in x+i,x_j

       As we are using point masses, this is just a sum over the 

    Args:
        masses (np.array): (nmasses) masses of objects
        coeffs (np.array): (ndimension, ncoeffs) x(t),y(t),z(t) position coefficients as a function of time
    Returns:
        second_mass_moment: second moment of the mass distribution
    """
    n_masses, n_dimensions, n_coeffs = np.shape(coeffs) 
    #using lists as I do not know the number of coeficcients after multiplying
    second_mass_moment = []#np.zeros((n_dimensions, n_dimensions, n_coeffs))

    for i in range(n_dimensions):
        second_mass_moment.append([])
        for j in range(n_dimensions):
            second_mass_moment[i].append([])
            for mass_ind in range(len(masses)):
                temp_moment = masses[mass_ind]*polynomial_dict[poly_type]["multiply"](coeffs[mass_ind, i], coeffs[mass_ind, j])
                if len(second_mass_moment[i][j]) == 0:
                    second_mass_moment[i][j] = temp_moment
                else:
                    second_mass_moment[i][j] += temp_moment
    
    second_mass_moment = np.array(second_mass_moment)

    if remove_trace:
        second_mass_moment = subtract_trace(second_mass_moment, poly_type=poly_type)

    return second_mass_moment

def compute_second_derivative_of_mass_moment(coeffs, poly_type="chebyshev"):
    """compute the second derivative of the second mass moment tensor

    Args:
        coeffs (_type_): (n_dimensions, n_dimensions, n_coeffs)

    Returns:
        _type_: second derivative of mass moment as coefficients
    """
    n_dimensions, _, n_coeffs = np.shape(coeffs)
    Iprime2_coeffs = [[[],[],[]], [[],[],[]],[[],[],[]]]#np.zeros((3,3,len(co_tensor[0][0]) - 2))

    for i in range(n_dimensions):
        for j in range(n_dimensions):
            Iprime2_coeffs[i][j] = polynomial_dict[poly_type]["derivative"](coeffs[i][j], m=2)

    return Iprime2_coeffs

def compute_projection_tensor(r=1, x=0, y=0, z=1):
    """compute the projection tensor delta_ij - n_i n_j
        n_i is the direction of propagation
        the default is to have this pointing along the z direction

    Args:
        r (int, optional): _description_. Defaults to 1.
        x (int, optional): _description_. Defaults to 0.
        y (int, optional): _description_. Defaults to 0.
        z (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: projection tensor
    """
    r = 1
    normx = x/r
    normy = y/r
    normz = z/r

    # projection tensor
    P = np.array([
        [1 - normx*normx, -normx*normy, -normx*normz],
        [-normx*normy, 1 - normy*normy, -normy*normz],
        [-normz*normx, -normz*normy, 1 - normz*normz]
    ])
    return P

def project_and_remove_trace(projection_tensor, coeffs, poly_type="chebyshev"):
    """project the tensor into the Transverse and subtract trace
      this is a slow way to do it !! speed it up!!
    Args:
        projection_tensor (_type_): _description_
        coeffs (_type_): _description_
    """


    Iprime2 = subtract_trace(coeffs, poly_type=poly_type)

    # compute the TT gauge strain tensor as projection tensor
    # see https://arxiv.org/pdf/gr-qc/0501041.pdf
    h_TT = []
    for i in range(3):
        h_TT.append([])
        for j in range(3):
            h_TT[i].append([])
            fact1 = []
            fact2 = []
            for k in range(3):
                for l in range(3):

                    t_fact1 = polynomial_dict[poly_type]["multiply"](
                        polynomial_dict[poly_type]["multiply"](
                            Iprime2[k,l], 
                            P[k,l]),
                        P[i,j])

                    t_fact2 += polynomial_dict[poly_type]["multiply"](
                        polynomial_dict[poly_type]["multiply"](
                            Iprime2[k,l], 
                            P[i,k]),
                        P[j,l])

                    if len(fact1) == 0:
                        fact1 = t_fact1
                    else:
                        fact1 += t_fact1
                    if len(fact2) == 0:
                        fact2 = t_fact2
                    else:
                        fact2 += t_fact2
                    #fact2 += Iprime2[k,l] * P[i,k] * P[j,l]

            if len(h_TT[i][j]) == 0:
                h_TT[i][i] = (fact2 - 0.5*fact1)
            else:
                h_TT[i][j] = (fact2 - 0.5*fact1)

    return np.array(h_TT)

def compute_hTT_coeffs(masses, coeffs, poly_type="chebyshev"):

    second_mass_moment = compute_second_mass_moment(masses, coeffs, remove_trace=True, poly_type=poly_type)
    projection_tensor = compute_projection_tensor()
    
    hTT = project_and_remove_trace(projection_tensor, second_mass_moment, poly_type=poly_type)

    return hTT


def generate_masses(n_masses: int) -> np.array:
    """generate masses 

    Args:
        n_masses (int): _description_

    Returns:
        np.array: _description_
    """

    masses = np.random.rand(n_masses)
    masses = masses/np.sum(masses)

    return masses

def antenna_pattern(alpha, delta, gpstime, detector="H1"):
    """compute the antenna pattern functions

    Args:
        alpha (_type_): _description_
        delta (_type_): _description_
        gpstime (_type_): _description_
        detector (str, optional): _description_. Defaults to "H1".

    Returns:
        _type_: _description_
    """
    time = lal.GreenwichMeanSiderealTime(gpstime)
    siteinfo = lalpulsar.GetSiteInfo(detector)
    am_plus, am_cross = lal.ComputeDetAMResponse(siteinfo.response, alpha, delta, 0.0, time)

    return am_plus, am_cross

def compute_strain(pols, detector="H1"):
    """the output strain for one sky position

    Args:
        pols (_type_): _description_

    Returns:
        _type_: _description_
    """
    # these are fixed for now so only have to be calculated once
    alpha, delta = np.pi, np.pi/2 # arbritrary values for now
    gpstime = 1381142123 # set to current time (when written)
    aplus, across = antenna_pattern(alpha, delta, gpstime, detector=detector)
    hplus, hcross = pols[0,0], pols[0,1]

    strain = aplus*hplus + across*hcross
    return strain

def compute_strain_from_coeffs(times, pols, detector="H1", poly_type="chebyshev"):
    """convert a set of coefficienct of hTT to a timeseries, theen compute the strain from hplus and hcross


    Args:
        times (np.array): times at which to evaluate the polynomial
        pols (_type_): (n_dimensions, n_dimensions, n_coeffs) coefficients for each polarisation

    Returns:
        _type_: _description_
    """
    # these are fixed for now so only have to be calculated once
    n_dimensions, n_dimensions, n_coeffs = np.shape(pols)
    hTT_timeseries = np.zeros((n_dimensions, n_dimensions, len(times)))
    for i in range(n_dimensions):
        for j in range(n_dimensions):
            hTT_timeseries[i,j] = polynomial_dict[poly_type]["val"](times, pols[i,j])

    strain = compute_strain(hTT_timeseries, detector=detector)

    return strain

def generate_data(n_data: int, chebyshev_order: int, n_masses:int, sample_rate: int, n_dimensions: int = 1, detectors=["H1"], window=False, return_windowed_coeffs=True, poly_type="chebyshev") -> np.array:
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

    strain_timeseries = np.zeros((n_data, len(detectors), sample_rate))

    times = np.arange(-1,1,2/sample_rate)

    random_coeffs = generate_random_coefficients(chebyshev_order, n_dimensions)

    if window != False:
        coeffs, win_coeffs = perform_window(times, random_coeffs, window, poly_type=poly_type)
    else:
        coeffs = random_coeffs


    if return_windowed_coeffs:  
        win_chebyshev_order = np.shape(coeffs)[0]
        acc_chebyshev_order = np.shape(coeffs)[0]
    else:
        win_chebyshev_order = np.shape(coeffs)[0]
        acc_chebyshev_order = chebyshev_order
    
    flattened_coeffs_mass = np.zeros((n_data, acc_chebyshev_order*n_masses*n_dimensions + n_masses))
    positions = np.zeros((n_data, n_masses, n_dimensions, len(times)))

    for data_index in range(n_data):

        masses = generate_masses(n_masses)

        all_dynamics = np.zeros((n_masses, n_dimensions, win_chebyshev_order))
        flattened_coeffs_mass[data_index, -n_masses:] = masses
        for mass_index in range(n_masses):

            random_coeffs = generate_random_coefficients(chebyshev_order, n_dimensions)
            # if windowing applied create coeffs which are windowed else just use the random coeffs
            if window:
                coeffs = window_coeffs(times, random_coeffs, win_coeffs, poly_type=poly_type)
            else:
                coeffs = random_coeffs

            # if we are just returning the random coeffs regardless of windowing create extra variable
            if not return_windowed_coeffs:
                random_flat_coeffs = np.ravel(random_coeffs)

            # also flatten (maybe windowed coeffs as needed for waveform generation)
            flat_coeffs = np.ravel(coeffs)

            if return_windowed_coeffs:
                flattened_coeffs_mass[data_index, acc_chebyshev_order*mass_index*n_dimensions:acc_chebyshev_order*n_dimensions*(mass_index+1)] = flat_coeffs
                all_dynamics[mass_index] = coeffs.T
            else:
                flattened_coeffs_mass[data_index, acc_chebyshev_order*mass_index*n_dimensions:acc_chebyshev_order*n_dimensions*(mass_index+1)] = random_flat_coeffs
                all_dynamics[mass_index] = coeffs.T

            positions[data_index, mass_index] = polynomial_dict[poly_type]["val"](times, coeffs)

        if n_dimensions == 1:
            strain_coeffs = generate_strain_coefficients(all_dynamics)
            strain_timeseries[data_index] = polynomial_dict[poly_type]["val"](times, strain_coeffs)
        elif n_dimensions == 2:
            temp_strain_timeseries = generate_2d_derivative(all_dynamics.reshape(chebyshev_order, n_dimensions), times)
            hplus = temp_strain_timeseries[0,0]
            hcross = temp_strain_timeseries[0,1]
            strain_timeseries[data_index][0] = hplus + hcross
        elif n_dimensions == 3:
            temp_strain_timeseries = compute_hTT_coeffs(masses, all_dynamics, poly_type=poly_type)

            for dind, detector in enumerate(detectors):
                strain_timeseries[data_index][dind] = compute_strain_from_coeffs(times, temp_strain_timeseries, detector, poly_type=poly_type)


    return times, flattened_coeffs_mass, strain_timeseries, acc_chebyshev_order, positions

if __name__ == "__main__":


    # TESTING code
    n_masses = 2
    chebyshev_order = 8
    n_dimensions = 3
    sample_rate = 32
    times = np.arange(-1,1,2/sample_rate)
    masses = generate_masses(n_masses)
    window = "tukey"

    #np.random.seed(123)

    if window == "tukey":
        win_coeffs = fit_cheby_to_tukey(times, alpha=0.1, order=30)
    elif window == "hann":
        win_coeffs = fit_cheby_to_hann(times, order=6)

    coeffs = generate_random_coefficients(chebyshev_order, n_dimensions)
    if window:
        coeffs = window_coeffs(times, coeffs, win_coeffs)

    acc_chebyshev_order = np.shape(coeffs)[0]
    

    all_dynamics = np.zeros(acc_chebyshev_order*n_dimensions)
    flattened_coeffs_mass = np.zeros((acc_chebyshev_order*n_masses*n_dimensions + n_masses))
    sep_dynamics = np.zeros((n_masses, acc_chebyshev_order, n_dimensions))
    flattened_coeffs_mass[-n_masses:] = masses

    for mass_index in range(n_masses):

        coeffs = generate_random_coefficients(chebyshev_order, n_dimensions)
        if window:
            coeffs = window_coeffs(times, coeffs, win_coeffs)
        flat_coeffs = np.ravel(coeffs)

        sep_dynamics[mass_index] = coeffs
        flattened_coeffs_mass[acc_chebyshev_order*mass_index*n_dimensions:acc_chebyshev_order*n_dimensions*(mass_index+1)] = flat_coeffs
        all_dynamics += masses[mass_index]*flat_coeffs

    temp_strain_timeseries = generate_3d_derivative(all_dynamics.reshape(acc_chebyshev_order, n_dimensions), times)
    
    alldyn = all_dynamics.reshape(acc_chebyshev_order, n_dimensions)

    fig, ax = plt.subplots(nrows = n_dimensions)
    for dim in range(n_dimensions):
        for ms in range(n_masses):
            dyn = polynomial_dict[poly_type]["val"](times, sep_dynamics[ms,:,dim])
            ax[dim].plot(times, dyn)

    fig.savefig("test_dyn.png")
    print(temp_strain_timeseries[0,0] + temp_strain_timeseries[1,1])
    print(temp_strain_timeseries[0,1] - temp_strain_timeseries[1,0])
    hplus = temp_strain_timeseries[0,0]
    hcross = temp_strain_timeseries[0,1]
    print(temp_strain_timeseries[:,:,0])

    fig, ax = plt.subplots()
    ax.plot(hplus)
    fig.savefig("testplot.png")
    
