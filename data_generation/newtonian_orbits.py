import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
import timeit
from ..basis.basis import basis

def newton_derivative(
    t, 
    x_posvels, 
    masses, 
    factor=1, 
    n_dimensions=3, 
    EPS=1e-13,
    second_scale=86400,
    mass_scale=1.989e30,
    distance_scale=1.4e11,
    G=6.67e-11,
    c=3e8
    ):
    """compute the derivative of the position and velocity

    Args:
        x_positions (_type_): _description_
        masses (_type_): _description_
    """
    n_masses = len(masses)

    # define some constants 
    G = 6.67e-11
    c = 3e8
    # compute G in gaussian gravitational units to prevent overflows
    # G * (s/day)/(m/Au) * Msun(kg)

    G_ggravunits = G * ((second_scale**2)/((distance_scale)**3)) * mass_scale
    c_ggravunits = c * ((second_scale**2)/(distance_scale))

    # reshape posvels to more useful shape
    x_posvels = x_posvels.reshape(n_masses, 2*n_dimensions)
    x_positions = x_posvels[:, 0:n_dimensions]
    x_vels = x_posvels[:, n_dimensions:2*n_dimensions]

    seps = np.zeros((n_masses, n_masses))

    x_derivative = np.zeros((n_masses, 2*n_dimensions))
    for i, mass_1 in enumerate(masses):
        x_derivative[i][0:n_dimensions] = x_vels[i]
        for j, mass_2 in enumerate(masses):
            if i == j: continue
            separation_cubed = np.sqrt(np.sum((x_positions[i] - x_positions[j])**2) + EPS)**3
            #seps[i,j] = separation_cubed
            diff = x_positions[j] - x_positions[i]
            x_derivative[i][n_dimensions:2*n_dimensions] += G_ggravunits*mass_2*diff/separation_cubed
         
        """
        # get all other masses but this one
        other_positions = np.delete(x_positions, i)
        other_vels = np.delete(x_vels, i)
        # compute separations
        rs = np.sqrt(np.sum((other_positions - x_positions[j])**2))
        """
    #energy_loss_term = x_derivative[i][3:6] * factor


    return x_derivative.flatten()

def newton_derivative_vect(
    t, 
    x_posvels, 
    masses, 
    factor=1, 
    n_dimensions=3, 
    EPS=1e-11,
    second_scale=86400,
    mass_scale=1.989e30,
    distance_scale=1.4e11,
    G=6.67e-11,
    c=3e8):
    """compute the derivative of the position and velocity

    Args:
        x_positions (_type_): _description_
        masses (_type_): _description_
    """
    n_masses = len(masses)

    # define some constants 
    # compute G in gaussian gravitational units to prevent overflows
    # G * (s/day)/(m/Au) * Msun(kg)
    G_ggravunits = G * ((second_scale**2)/((distance_scale)**3)) * mass_scale
    c_ggravunits = c * ((second_scale**2)/(distance_scale))

    x_posvels = x_posvels.reshape(n_masses, 2*n_dimensions)
    x_derivative = np.zeros((n_masses, 2*n_dimensions))

    # compute separations of each object and the absolute distance
    diff_xyz = x_posvels[:,0:3,None].T - x_posvels[:,0:3,None] # Nx3xxN matrix
    #inv_r_cubed = (np.sum(diff_xyz**2, axis=1) + EPS)**(-1.5) # NxN matrix
    inv_r_cubed = (np.einsum("ijk,ijk->ik", diff_xyz, diff_xyz) + EPS)**-1.5
    #print("invdiff", np.sum(diff_xyz**2, axis=1) - inv_r_cubed2)

    # below two lines are the same
    # take matrix multiplication of masses with last dimensions
    #acceleration = G*(dxyz * inv_r_cubed) @ masses
    #acceleration = np.einsum("ijk,k", G*(dxyz * inv_r_cubed), masses)

    x_derivative[:, 0:3] = x_posvels[:, 3:6]
    #div_r_cubed = np.einsum("ijk, ik->ijk", G_ggravunits*diff_xyz, inv_r_cubed)
    x_derivative[:, 3:6] = np.einsum("ijk, ik, k->ij", G_ggravunits*diff_xyz, inv_r_cubed, masses)
    #x_derivative[:, 3:6] = np.einsum("ijk,k->ij", div_r_cubed, masses)

    return x_derivative.flatten()


def get_initial_conditions(n_masses, n_dimensions):
    """get positions and velocities
    position in m/1e5
    velocity in m/1e5/day

    Args:
        n_masses (_type_): _description_
        n_dimensions (_type_): _description_

    Returns:
        _type_: _description_
    """
    return np.random.uniform(-1,1,size=(n_masses, 2*n_dimensions))

def get_masses(n_masses):

    masses = np.random.uniform(0,1,size=n_masses)

    return masses/np.sum(masses)

def interpolate_positions(old_times, new_times, positions):
    """Interpolate between points """
    interp_dict = np.zeros((positions.shape[0], len(new_times)))
    for object_ind in range(positions.shape[0]):
        #print(np.shape(positions[object_ind]))
        interp = interp1d(old_times, positions[object_ind], kind="cubic")
        interp_dict[object_ind] = interp(new_times)
    return interp_dict

def solve_ode(
    n_masses=2, 
    n_dimensions=3, 
    n_samples=128
    ):


    second = 1./(24*3600)
    times = np.linspace(0,2*second,n_samples)

    # scale to 1 year
    second_scale = 86400
    # between 0 and au/100
    distance_scale = 1.4e11*2e-4
    # between 0 and 100 solar masses
    mass_scale = 1.989e30*100

    initial_conditions = get_initial_conditions(n_masses, n_dimensions)
    initial_conditions[:,3:6] *= 1e4
    masses = get_masses(n_masses)

    ode = lambda t, x: newton_derivative_vect(
        t, 
        x, 
        masses=masses, 
        factor=1, 
        n_dimensions=n_dimensions,
        second_scale=second_scale,
        distance_scale=distance_scale,
        mass_scale=mass_scale)

    outputs = solve_ivp(
        ode,
        [min(times) - second, max(times) + second], 
        initial_conditions.flatten(), 
        tfirst=True,
        method="LSODA",
        rtol = 1e-5)
    
    if max(outputs.t) < max(times):
        outputs.t = np.append(outputs.t, max(times)+0.1)
        outputs.y = np.append(outputs.y.T, outputs.y[:, -1:].T, axis=0).T

    # y is shape (ntimes, )
    positions = outputs.y[:, :]
    interp_positions = interpolate_positions(outputs.t, times, positions).T
    interp_positions = interp_positions.reshape(len(times), n_masses, 2*n_dimensions)[:,:,:3]

    interp_positions = interp_positions - np.mean(interp_positions, axis=(0, 1))[None,None,:]

    return times, interp_positions, masses


def generate_newton_data(
    n_data: int, 
    basis_order: int, 
    n_masses:int, 
    sample_rate: int, 
    n_dimensions: int = 3, 
    detectors=["H1"], 
    window="none", 
    return_windowed_coeffs=True, 
    basis_type="chebyshev",) -> np.array:
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

    ntimeseries = [0, 1, 3, 6, 10]

    strain_timeseries = np.zeros((n_data, len(detectors), sample_rate))

    times = np.arange(-1,1,2/sample_rate)

    times, coeffs, masses = solve_ode(
        n_masses=n_masses, 
        n_dimensions=n_dimensions, 
        n_samples=len(times))

    print(np.shape(coeffs))


    if return_windowed_coeffs:  
        win_basis_order = np.shape(coeffs)[0]
        acc_basis_order = np.shape(coeffs)[0]
    else:
        win_basis_order = np.shape(coeffs)[0]
        acc_basis_order = basis_order

        if basis_type == "fourier":
            # plus 2 as plus 1 in the real and imaginary in coeff gen
            # this is so can get back to order is 1/2 n_samples
            acc_basis_order += 2

    output_coeffs_mass = np.zeros((n_data, acc_basis_order*n_masses*n_dimensions + n_masses))
    positions = np.zeros((n_data, n_masses, n_dimensions, len(times)))
    if basis_type == "fourier":
        all_dynamics = np.zeros((n_data, n_masses, n_dimensions, int(0.5*acc_basis_order)), dtype=dtype)
    else:
        all_dynamics = np.zeros((n_data, n_masses, n_dimensions, acc_basis_order), dtype=dtype)

    for data_index in range(n_data):

        if data_index %2 == 0:
            print(data_index)
        times, temp_position, masses = solve_ode(
            n_masses=n_masses, 
            n_dimensions=n_dimensions, 
            n_samples=len(times))
        # position shape (n_samples, n_masses, n_dimensions)

        temp_output_coeffs = np.zeros((n_masses, n_dimensions, acc_basis_order))
        for mass_index in range(n_masses):
            if basis_type=="fourier":
                temp_dyn = basis[basis_type]["fit"](
                    times,
                    temp_position[:,mass_index,:].T,
                    basis_order
                    )
            else:
                temp_dyn = basis[basis_type]["fit"](
                    times,
                    temp_position[:,mass_index,:],
                    basis_order-1
                    ).T
            all_dynamics[data_index, mass_index] = temp_dyn

            if basis_type == "fourier":
                temp_dyn = torch.view_as_real(torch.from_numpy(temp_dyn))
                tdshape = temp_dyn.shape
                temp_dyn = temp_dyn.flatten(start_dim=1)#temp_dyn.reshape(tdshape[0], tdshape[1]*tdshape[2])
           
            temp_output_coeffs[mass_index] = temp_dyn

        output_coeffs_mass[data_index] = np.append(temp_output_coeffs.flatten(), masses)

        positions[data_index] = np.transpose(temp_position, (1, 2, 0))

        if n_dimensions == 3:
            temp_strain_timeseries = compute_hTT_coeffs(masses, all_dynamics[data_index], basis_type=basis_type)

            for dind, detector in enumerate(detectors):
                strain = compute_strain_from_coeffs(times, temp_strain_timeseries, detector, basis_type=basis_type)
                window = signal.windows.tukey(np.shape(strain)[-1], alpha=0.5)
                # shape (n_dims, n_times)
                #strain_timeseries[data_index][dind] = strain * window[None, :]
                strain_timeseries[data_index][dind] = strain

        else:
            raise Exception("Only runs for three dimensional data")

    return times, output_coeffs_mass, strain_timeseries, acc_basis_order, positions, all_dynamics


if __name__ == "__main__":

    
    times, outputs, masses = solve_ode(
        n_samples=512
    )

    #print(outputs)

    fig, ax = plt.subplots()

    ax.plot(outputs[:, 0, 0], outputs[:, 0, 1], markersize = masses[0]*1e-2, marker="o")
    ax.plot(outputs[:, 1, 0], outputs[:, 1, 1], markersize = masses[1]*1e-2, marker="o")
    ax.plot(outputs[0, 0, 0], outputs[0, 0, 1], markersize = masses[0]*1, marker="o", color="k")
    ax.plot(outputs[0, 1, 0], outputs[0, 1, 1], markersize = masses[1]*1, marker="o", color="k")

    fig.savefig("./test_ode.png")

    
    second = 1./(24*3600)
    times = np.linspace(0,2*second,128)

    initial_conditions = get_initial_conditions(2, 3)
    initial_conditions[:,3:6] *= 1e4
    masses = get_masses(2)*4e3

    # scale to 1 year
    second_scale = 86400
    # between 0 and au/100
    distance_scale = 1.4e11*1e-4
    # between 0 and 100 solar masses
    mass_scale = 1.989e30*100

    start_time = time.time()
    der1 = lambda: newton_derivative_vect(
        times[0], 
        initial_conditions.flatten(), 
        masses,
        n_dimensions=3,
        second_scale=second_scale,
        distance_scale=distance_scale,
        mass_scale=mass_scale)
    print("vectorised", time.time() - start_time)

    start_time = time.time()
    der2 = lambda: newton_derivative(
        times[0], 
        initial_conditions.flatten(), 
        masses,
        n_dimensions=3,
        second_scale=second_scale,
        distance_scale=distance_scale,
        mass_scale=mass_scale)

    print("loops", time.time() - start_time)

    t1 = timeit.timeit(der1, number=10000)
    t2 = timeit.timeit(der2, number=10000)
    print("vec,loop", t1, t2 )

    #print(der1 - der2)
    