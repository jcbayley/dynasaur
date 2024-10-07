import numpy as np
from dynasaur.basis_functions import basis
import lal
import lalpulsar

def generate_strain_coefficients(coeffs: np.array, basis_type="chebyshev", duration=2) -> np.array:
    """
    Generate the strain coefficients from the dynamics coefficients
    Args:
        coeffs (np.array): _description_

    Returns:
        np.array: _description_
    """

    squares_chebyshev = basis[basis_type]["power"](coeffs, 2)
    diff_chebyshev = basis[basis_type]["derivative"](squares_chebyshev, m=2, duration=duration)

    return diff_chebyshev

def subtract_trace(coeffs, basis_type="chebyshev"):
    """subtract the trace from a 3d tensor

    Args:
        coeffs (_type_): _description_

    Returns:
        _type_: _description_
    """
    # sum diagonal to compute trace
    n_dimensions, n_dimensions, n_coeffs = np.shape(coeffs)
    trace = basis[basis_type]["add"](
        basis[basis_type]["add"](
            coeffs[0,0], 
            coeffs[1,1]), 
        coeffs[2,2])
    # divide by three the subtract from diagonals
    factor = basis[basis_type]["multiply"](trace, 1./3)
    for i in range(n_dimensions):
        subbed = basis[basis_type]["subtract"](coeffs[i,i], factor)
        coeffs[i,i, :len(subbed)] = subbed
    return coeffs

def compute_second_mass_moment(
    masses, 
    coeffs, 
    remove_trace = False, 
    basis_type="chebyshev"):
    """Performs integral over density in x+i,x_j

       As we are using point masses, this is just a sum over the 

    Args:
        masses (np.array): (nmasses) masses of objects
        coeffs (np.array): (nmasses, ndimension, ncoeffs) x(t),y(t),z(t) position coefficients as a function of time
    Returns:
        second_mass_moment: second moment of the mass distribution
    """
    n_masses, n_dimensions, n_coeffs = np.shape(coeffs) 
    #using lists as I do not know the number of coeficcients after multiplying
    # find out what the shape will be with a quick test

    new_coeff_shape = n_coeffs
    
    second_mass_moment = np.zeros((n_dimensions, n_dimensions, new_coeff_shape), dtype=basis[basis_type]["dtype"])

    for i in range(n_dimensions):
        #second_mass_moment.append([])
        for j in range(n_dimensions):
            #second_mass_moment[i].append([])
            for mass_ind in range(len(masses)):
                mult = masses[mass_ind]*basis[basis_type]["multiply"](coeffs[mass_ind, i], coeffs[mass_ind, j])
                if len(mult) > new_coeff_shape:
                    second_mass_moment.resize((n_dimensions, n_dimensions, len(mult)), refcheck=False)
    
                second_mass_moment[i,j, :len(mult)] += basis[basis_type]["multiply"](mult, masses[mass_ind])
                """
                temp_moment = masses[mass_ind]*basis[basis_type]["multiply"](coeffs[mass_ind, i], coeffs[mass_ind, j])
                if len(second_mass_moment[i][j]) == 0:
                    second_mass_moment[i][j] = temp_moment
                else:
                    second_mass_moment[i][j] += temp_moment
                """
    second_mass_moment = np.array(second_mass_moment)

    if remove_trace:
        second_mass_moment = subtract_trace(second_mass_moment, basis_type=basis_type)

    return second_mass_moment


def compute_derivative_of_mass_moment(coeffs, order=2, basis_type="chebyshev", duration=2):
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
            Iprime2_coeffs[i][j] = basis[basis_type]["derivative"](coeffs[i][j], m=2, duration=duration)

    return np.array(Iprime2_coeffs)

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

def project_and_remove_trace(projection_tensor, coeffs, basis_type="chebyshev"):
    """project the tensor into the Transverse and subtract trace
      this is a slow way to do it !! speed it up!!
    Args:
        projection_tensor (_type_): _description_
        coeffs (_type_): _description_
    """

    n_dimensions, n_dimensions, n_coeffs = np.shape(coeffs)
    Iprime2 = subtract_trace(coeffs, basis_type=basis_type)

    # compute the TT gauge strain tensor as projection tensor
    # see https://arxiv.org/pdf/gr-qc/0501041.pdf

    n_new_coeffs = 0

    # pre estimate the max num of coefficients

    temp_mult = basis[basis_type]["multiply"](
        basis[basis_type]["multiply"](
            Iprime2[0,0], 
            projection_tensor[0,0]),
        projection_tensor[0,0])

    h_TT = np.zeros((n_dimensions, n_dimensions, len(temp_mult)), dtype=basis[basis_type]["dtype"])

    for i in range(3):
        #h_TT.append([])
        for j in range(3):
            #h_TT[i].append([])
            fact1 = np.zeros(len(temp_mult), dtype=basis[basis_type]["dtype"])
            fact2 = np.zeros(len(temp_mult), dtype=basis[basis_type]["dtype"])
            for k in range(3):
                for l in range(3):

                    t_fact1 = basis[basis_type]["multiply"](
                        basis[basis_type]["multiply"](
                            Iprime2[k,l], 
                            projection_tensor[k,l]),
                        projection_tensor[i,j])

                    t_fact2 = basis[basis_type]["multiply"](
                        basis[basis_type]["multiply"](
                            Iprime2[k,l], 
                            projection_tensor[i,k]),
                        projection_tensor[j,l])
                    
                    if len(t_fact1) == 1:
                        t_fact1 = np.repeat(t_fact1, len(fact1))
                    if len(t_fact2) == 1:
                        t_fact2 = np.repeat(t_fact2, len(fact2))

                    fact1[:len(t_fact1)] += t_fact1
                    fact2[:len(t_fact2)] += t_fact2
            
            h_TT[i,j] = fact2 - 0.5*fact1
            """
                    if len(fact1) == 0 or len(fact1) == 1:
                        fact1 = t_fact1
                    else:
                        fact1 += t_fact1
                    if len(fact2) == 0 or len(fact2) == 1:
                        fact2 = t_fact2
                    else:
                        fact2 += t_fact2
                    #fact2 += Iprime2[k,l] * P[i,k] * P[j,l]

            if len(h_TT[i][j]) == 0:
                h_TT[i][i] = (fact2 - 0.5*fact1)
            else:
                h_TT[i][j] = (fact2 - 0.5*fact1)
            """
    return np.array(h_TT)

def compute_hTT_coeffs(masses, coeffs, basis_type="chebyshev", duration=2):
    """compute the htt coefficients for polynomial

    Args:
        masses (_type_): _description_
        coeffs (_type_): _description_
        basis_type (str, optional): _description_. Defaults to "chebyshev".

    Returns:
        _type_: _description_
    """
    second_mass_moment = compute_second_mass_moment(masses, coeffs, remove_trace=True, basis_type=basis_type)
    second_mass_moment_derivative = compute_derivative_of_mass_moment(second_mass_moment, order=2, basis_type=basis_type, duration=duration)
    projection_tensor = compute_projection_tensor()
    hTT = project_and_remove_trace(projection_tensor, second_mass_moment_derivative, basis_type=basis_type)

    return hTT

def compute_energy_loss(times, masses, coeffs, basis_type="chebyshev", duration=2):
    """compute the energy as a function of time

    Args:
        masses (_type_): _description_
        coeffs (_type_): _description_
        basis_type (str, optional): _description_. Defaults to "chebyshev".

    Returns:
        _type_: _description_
    """
    second_mass_moment = compute_second_mass_moment(masses, coeffs, remove_trace=True, basis_type=basis_type)
    second_mass_moment_derivative = compute_derivative_of_mass_moment(second_mass_moment, order=3, basis_type=basis_type, duration=duration)
    projection_tensor = compute_projection_tensor()
    
    Ider3 = project_and_remove_trace(projection_tensor, second_mass_moment_derivative, basis_type=basis_type)

    n_dimensions, n_dimensions, n_coeffs = np.shape(Ider3)
  
    Ider3_timeseries = basis[basis_type]["val"](times, Ider3)
    """
    Ider3_timeseries = np.zeros((n_dimensions, n_dimensions, len(times)))
    for i in range(n_dimensions):
        for j in range(n_dimensions):
            Ider3_timeseries[i,j] = basis[basis_type]["val"](times, Ider3[i,j])
    """
  
    energy = np.sum(Ider3_timeseries**2, axis = (0,1))

    return energy


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

def compute_strain(pols, detector="H1", alpha=np.pi, delta=np.pi/2, gpstime=1381142123):
    """the output strain for one sky position

    Args:
        pols (_type_): _description_

    Returns:
        _type_: _description_
    """
    # these are fixed for now so only have to be calculated once
    #alpha, delta = np.pi, np.pi/2 # arbritrary values for now
    #gpstime = 1381142123 # set to current time (when written)
    hplus, hcross = pols[0,0], pols[0,1]
    if detector == "hplus":
        strain = hplus
    elif detector == "hcross":
        strain = hcross
    else:
        aplus, across = antenna_pattern(alpha, delta, gpstime, detector=detector)
        strain = aplus*hplus + across*hcross
    return strain

def compute_strain_from_coeffs(times, pols, detectors=["H1"], basis_type="chebyshev", sky_position=(np.pi, np.pi/2)):
    """convert a set of coefficienct of hTT to a timeseries, theen compute the strain from hplus and hcross


    Args:
        times (np.array): times at which to evaluate the polynomial
        pols (_type_): (n_dimensions, n_dimensions, n_coeffs) coefficients for each polarisation
        sky : tuple of sky position

    Returns:
        _type_: _description_
    """
    # these are fixed for now so only have to be calculated once
    n_dimensions, n_dimensions, n_coeffs = np.shape(pols)
    """
    hTT_timeseries = np.zeros((n_dimensions, n_dimensions, len(times)))
    for i in range(n_dimensions):
        for j in range(n_dimensions):
            hTT_timeseries[i,j] = basis[basis_type]["val"](times, pols[i,j])
    """
    # input to val should be (nbasis, ndim, ndim) as val evaluated over 1st dimension
    # the output of val should switch back to having time dimension last
    hTT_timeseries = basis[basis_type]["val"](times, pols)

    strain_timeseries = np.zeros((len(detectors), len(times)))
    for dind, detector in enumerate(detectors):
        strain_timeseries[dind] = compute_strain(hTT_timeseries, detector=detector, alpha=sky_position[0], delta=sky_position[1])

    return strain_timeseries

def compute_hplus_hcross(masses, x, xdot, xddot):
    """compute the plus and cross polarisations

    Args:
        masses (_type_): (N_masses)
        x (_type_): (N_masses, N_dimensions, N_times)
        xdot (_type_): (N_masses, N_dimensions, N_times)
        xddot (_type_): (N_masses, N_dimensions, N_times)

    Returns:
        tuple: hplus, hcross
    """
    norm_factor = 1

    plus_t1 = masses[0]*(
        xddot[:,0,0]*x[:,0,0] 
        + xdot[:,0,0]**2 
        - xddot[:,0,1]*x[:,0,1] 
        - xdot[:,0,1]**2)
    plus_t2 = masses[1]*(
        xddot[:,1,0]*x[:,1,0] 
        + xdot[:,1,0]**2 
        - xddot[:,1,1]*x[:,1,1] 
        - xdot[:,1,1]**2)
    hplus = norm_factor*(plus_t1 + plus_t2)

    cross_t1 = masses[0]*(
        xddot[:,0,0]*x[:,0,1] 
        + 2*xdot[:,0,0]*xdot[:,0,1] 
        + x[:,0,0]*xddot[:,0,1])
    cross_t2 = masses[1]*(
        xddot[:,1,0]*x[:,1,1] 
        + 2*xdot[:,1,0]*xdot[:,1,1] 
        + x[:,1,0]*xddot[:,1,1])
    hcross = 4*norm_factor*(cross_t1 + cross_t2)
    return hplus, hcross

def dynamics_to_hplus_hcross(times, masses, basis_dynamics, basis_type="fourier"):

    norm_factor = 1

    fx = basis_dynamics
    fxdot = basis[basis_type]["derivative"](basis_dynamics, m=1)
    fxddot = basis[basis_type]["derivative"](basis_dynamics, m=2)

    fx[np.isinf(fx)] = 0
    fxdot[np.isinf(fxdot)] = 0
    fxddot[np.isinf(fxddot)] = 0

    x = basis[basis_type]["val"](times, fx)
    xdot = basis[basis_type]["val"](times, fxdot)
    xddot = basis[basis_type]["val"](times, fxddot)

    hplus, hcross = compute_hplus_hcross(masses, x, xdot, xddot)

    pols = np.array([[hplus, hcross], [hcross, -hplus]])

    return hplus, hcross, x, xdot, xddot

def get_waveform(
    times, 
    norm_masses, 
    basis_dynamics, 
    detectors, 
    basis_type="chebyshev",
    compute_energy=False,
    sky_position=(np.pi, np.pi/2)
    ):
    

    duration = np.max(times) - np.min(times)
    
    strain_coeffs = compute_hTT_coeffs(norm_masses, basis_dynamics, basis_type=basis_type, duration=duration)

    if compute_energy:
        energy = compute_energy_loss(times, norm_masses, basis_dynamics, basis_type=basis_type, duration=duration)
    else:
        energy = None

    strain_timeseries = compute_strain_from_coeffs(times, strain_coeffs, detectors=detectors, basis_type=basis_type, sky_position=sky_position)
    """
    strain_timeseries = np.zeros((len(detectors), len(times)))
    for dind, detector in enumerate(detectors):
        strain = compute_strain_from_coeffs(times, strain_coeffs, detector=detector, basis_type=basis_type)
        #strain = compute_strain(temp_strain_timeseries, detector, basis_type=basis_type)
        print("strain", np.shape(strain))
        strain_timeseries[dind] = strain
    """
    return strain_timeseries, energy

def generate_outputs(
    times,
    masses,
    dynamics, 
    basis_type="chebyshev"):

    duration = np.max(times) - np.min(times)
    if n_dimensions == 1:
        strain_coeffs = generate_strain_coefficients(all_dynamics[data_index], duration)
        strain_timeseries[data_index] = basis[basis_type]["val"](times, strain_coeffs)
    elif n_dimensions == 2:
        temp_strain_timeseries = generate_2d_derivative(all_dynamics[data_index].reshape(basis_order, n_dimensions), times)
        hplus = temp_strain_timeseries[0,0]
        hcross = temp_strain_timeseries[0,1]
        strain_timeseries[data_index][0] = hplus + hcross
    elif n_dimensions == 3:
        temp_strain_timeseries = compute_hTT_coeffs(masses, all_dynamics[data_index], basis_type=basis_type, duration=duration)

        for dind, detector in enumerate(detectors):
            strain_timeseries[data_index][dind] = compute_strain_from_coeffs(times, temp_strain_timeseries, detector, basis_type=basis_type)

    return strain_timeseries

def get_time_dynamics(
    coeff_samples, 
    times, 
    basis_type="chebyshev"):
    """get the dynamics of the system from polynomial cooefficients and masses

    Args:
        coeffmass_samples (np.array): (Nmasses, Ndimensions, Ncoeffs) samples of the coefficients and masses
        mass_samples (np.array): (Nmasses)
        times (_type_): times when to evaluate the polynomial
        n_masses (_type_): how many masses 
        basis_order (_type_): order of the polynomimal
        n_dimensions (_type_): how many dimensions 

    Returns:
        tuple: (coefficients, masses, timeseries)
    """
    n_masses, n_dimensions, n_coeffs = np.shape(coeff_samples)
    # has to be reordered due to how the np polynomial computes val on a matrix
    tseries = basis[basis_type]["val"](times, coeff_samples)
    """
    tseries = np.zeros((n_masses, n_dimensions, len(times)))
    for mass_index in range(n_masses):
        for dim_index in range(n_dimensions):
            tseries[mass_index, dim_index] = basis[basis_type]["val"](times, coeff_samples[mass_index, dim_index, :])
    """
    return tseries

