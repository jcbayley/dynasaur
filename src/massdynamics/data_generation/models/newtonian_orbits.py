import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
import timeit
from massdynamics.basis_functions import basis
from massdynamics.data_generation import orbits_functions, data_processing

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
        x_positions (_type_): (Nmasses*6, )
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


def get_masses(n_masses):

    masses = np.random.uniform(0,1,size=n_masses)

    return masses/np.sum(masses)

def interpolate_positions(old_times, new_times, positions):
    """Interpolate between points over dimension 1 """
    interp_dict = np.zeros((positions.shape[0], len(new_times)))
    for object_ind in range(positions.shape[0]):
        #print(np.shape(positions[object_ind]))
        interp = interp1d(old_times, positions[object_ind], kind="cubic")
        interp_dict[object_ind] = interp(new_times)
    return interp_dict

def too_close_event(t, x, n_masses=2, n_dimensions=3):

    x = x.reshape(n_masses, 2*n_dimensions)
    x_derivative = np.zeros((n_masses, 2*n_dimensions))

    # compute separations of each object and the absolute distance
    diff_xyz = x[:,0:3,None].T - x[:,0:3,None] # Nx3xxN matrix
    #inv_r_cubed = (np.sum(diff_xyz**2, axis=1) + EPS)**(-1.5) # NxN matrix
    r_cubed = (np.einsum("ijk,ijk->ik", diff_xyz, diff_xyz))**0.5

    if np.any(r_cubed < 1e-4):
        return r_cubed

def generate_kepler_orbit(times, semi_major_axes, eccentricities, inclinations, G, M):
    
    # Generate initial conditions for Keplerian orbits
    positions = []
    velocities = []
    num_orbits = len(semi_major_axes)

    for i in range(num_orbits):
        # Semi-major axis (in meters)
        a = semi_major_axes[i]

        # Eccentricity
        e = eccentricities[i]

        # Inclination (angle in radians)
        inclination = inclinations[i]

        # Orbital period (Kepler's third law)
        period = np.sqrt(4 * np.pi**2 * a**3 / (G * M))

        # Generate an array of time points covering one orbit
        #t = 0#np.linspace(0, period, 1000)

        # Calculate mean anomaly
        mean_anomaly = 2 * np.pi * times / period

        # Solve Kepler's equation for eccentric anomaly
        eccentric_anomaly = np.arctan2(np.sqrt(1 - e**2) * np.sin(mean_anomaly), e + np.cos(mean_anomaly))

        # Calculate true anomaly
        true_anomaly = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(eccentric_anomaly / 2), np.sqrt(1 - e) * np.cos(eccentric_anomaly / 2))

        # Generate polar coordinates in the orbital plane
        r = a * (1 - e**2) / (1 + e * np.cos(true_anomaly))
        theta = true_anomaly + inclination

        # Convert polar coordinates to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.zeros(np.shape(x))

        # Calculate radial velocity
        v_r = np.sqrt(G * M * (2 / r - 1 / a))

        # Calculate tangential velocity
        #v_t = np.sqrt(2 * G * M / r - G * M / a)
        v_t = np.sqrt(2 * G * M / r)

        # Convert polar velocities to Cartesian velocities
        vx = v_r * np.cos(theta) - v_t * np.sin(theta)
        vy = v_r * np.sin(theta) + v_t * np.cos(theta)
        vz = np.zeros(np.shape(vx))

        # Append positions and velocities
        positions.append(np.array([x, y, z]))
        velocities.append(np.array([vx, vy, vz]))

    return positions, velocities
    
def get_initial_positions_velocities(n_masses, n_dimensions, position_scale, velocity_scale):

    
    initial_positions = np.random.uniform(
        -1,
        1, 
        size=(n_masses, n_dimensions)
        )*position_scale
    initial_velocities = np.random.uniform(
        -1,
        1, 
        size=(n_masses, n_dimensions)
        )*velocity_scale
    """
    initial_positions = np.ones((n_masses, n_dimensions))*position_scale
    initial_positions[0] *= -1
    initial_positions[:,1:] *= 0
    initial_velocities = np.ones((n_masses, n_dimensions))*velocity_scale
    initial_velocities[0] *= -1
    initial_velocities[:,2] *= 0
    initial_velocities[:,0] *= 0
    """
    return initial_positions, initial_velocities

def get_initial_conditions(
    times, 
    G, 
    n_masses, 
    n_dimensions,
    position_scale, 
    mass_scale, 
    velocity_scale,
    data_type = "kepler"):
    """return initial positions, velocities and masses
       these should be in units of 
       mass: kg
       positions: m
       velocities: m/s

    Args:
        times (): _description_
        G (_type_): _description_
        M (_type_): _description_
        n_masses (_type_): _description_
        n_dimensions (_type_): _description_
        position_scale (_type_): _description_
        mass_scale (_type_): _description_
        velocity_scale (_type_): _description_
        initial_type (str, optional): _description_. Defaults to "kepler".

    Returns:
        _type_: _description_
    """
    if data_type == "equalmass_3d":
        masses = np.array([0.5,0.5])*mass_scale*1e-2
        initial_positions, initial_velocities = get_initial_positions_velocities(n_masses, n_dimensions, position_scale, velocity_scale)
    elif data_type == "kepler_fixedperiod":
        M = 1e30
        duration = np.max(times) - np.min(times)
        n_samples = len(times)
        period = duration
        min_period = 2.*duration/n_samples

        masses = np.array([M, np.random.uniform(1e-5*M, 1e-3*M)])

        semi_major_axes = (G*np.sum(masses)/(4*np.pi**2) * period**2)**(1/3)
        eccentricities = np.random.uniform(0.0,0.9,size=1)
        inclinations = np.random.uniform(0.0,2*np.pi,size=1)

        initial_positions, initial_velocities = generate_kepler_orbit(
            times, 
            [semi_major_axes], 
            eccentricities, 
            inclinations, 
            G, 
            M)


    elif data_type == "kepler":
        M = 1e30
        duration = np.max(times) - np.min(times)
        n_samples = len(times)
        period = duration
        min_period = 2.*duration/n_samples

        masses = np.array([M, np.random.uniform(1e-5*M, 1e-3*M)])

        semi_major_axes = (G*np.sum(masses)/(4*np.pi**2) * period**2)**(1/3)
        eccentricities = np.random.uniform(0.0,0.9,size=1)
        inclinations = np.random.uniform(0.0,2*np.pi,size=1)

        initial_positions, initial_velocities = generate_kepler_orbit(
            times, 
            semi_major_axes, 
            eccentricities, 
            inclinations, 
            G, 
            M)
        

    return masses, initial_positions, initial_velocities

def resample_initial_conditions(
    times, 
    G, 
    n_masses, 
    n_dimensions, 
    position_scale, 
    mass_scale, 
    velocity_scale,
    data_type = "kepler_fixedperiod"
    ):

    masses, initial_positions, initial_velocities = get_initial_conditions(
        times, 
        G, 
        n_masses, 
        n_dimensions, 
        position_scale, 
        mass_scale, 
        velocity_scale,
        data_type = data_type)


    oenergy = orbits_functions.orbital_energy(
        masses[0], 
        masses[1], 
        initial_positions[0], 
        initial_positions[1], 
        initial_velocities[0], 
        initial_velocities[1], 
        G)

    #print(oenergy)
    # set energy so orbit is bound, 
    
    n_resampled = 0
    while oenergy > 0.0:
        #initial_positions, initial_velocities = get_initial_positions_velocities(n_masses, n_dimensions, position_scale, velocity_scale)
        masses, initial_positions, initial_velocities = get_initial_conditions(
            times, 
            G, 
            M, 
            n_masses, 
            n_dimensions, 
            position_scale, 
            mass_scale, 
            velocity_scale,
            data_type = data_type)

        oenergy = orbits_functions.orbital_energy(
            masses[0], 
            masses[1], 
            initial_positions[0], 
            initial_positions[1], 
            initial_velocities[0], 
            initial_velocities[1], 
            G)
        
        n_resampled += 1
    #print("N_resampled: ", n_resampled)

    return masses, initial_positions, initial_velocities

def solve_ode(
    times, 
    masses,
    initial_positions,
    initial_velocities,
    G=6.67e-11,
    c=3e8,
    n_samples=128):

    # scalings for the ode solver
    # scale to 1 year
    second_scale = 86400
    # scale to 1 au
    distance_scale = 1.4e11
    #  solar masses
    mass_scale = 1.989e30

    #position_scale = 2*distance_scale                             # in m
    #velocity_scale = np.sqrt(2*G*mass_scale/distance_scale)*1e-1       # in m/s

    G_ggravunits = G * ((second_scale**2)/((distance_scale)**3)) * mass_scale
    c_ggravunits = c * ((second_scale**2)/(distance_scale))

    # rescale positions and velocities and masses
    ode_initial_positions = initial_positions/distance_scale                # now in AU
    ode_initial_velocities = initial_velocities*second_scale/distance_scale # now in AU/day
    ode_masses = masses/mass_scale                                          # now in solar masses

    initial_positions = data_processing.subtract_center_of_mass(initial_positions[:,:,np.newaxis], masses)[:,:,0]

    #scaled_initial_velocities = initial_velocities/velocity_scale
    #scaled_initial_positions = initial_positions/position_scale
    #scaled_masses = masses/mass_scale
    #scaled_masses = masses/np.sum(masses)

    #print(ode_initial_positions, initial_positions)
    #print(ode_initial_velocities, initial_velocities)
    #print(ode_masses, masses)

    initial_conditions = np.concatenate([ode_initial_positions, ode_initial_velocities], axis=-1)

    #print(initial_conditions)
    ode = lambda t, x: newton_derivative(
        t, 
        x, 
        masses=ode_masses, 
        factor=1, 
        n_dimensions=n_dimensions,
        second_scale=second_scale,
        distance_scale=distance_scale,
        mass_scale=mass_scale)
    
    #too_close_event.terminal = True

    outputs = solve_ivp(
        ode,
        t_span=[min(times), max(times)], 
        y0=initial_conditions.flatten(), 
        tfirst=True,
        method="LSODA",
        rtol=1e-6,
        atol=1e-6)
    
    """
    if max(outputs.t) < max(times):
        outputs.t = np.append(outputs.t, max(times)+0.1)
        outputs.y = np.append(outputs.y.T, outputs.y[:, -1:].T, axis=0).T
    """
    # y is shape (nvals, ntimes, )

    positions = outputs.y.reshape(n_masses, 2*n_dimensions, len(outputs.t))[:,:3] # get positions only
    positions = positions.reshape(n_masses*n_dimensions, len(outputs.t))

    ode_interp_positions = interpolate_positions(outputs.t, times, positions).T # ntimes, nvals
    ode_interp_positions = ode_interp_positions.reshape(len(times), n_masses, n_dimensions)
    ode_interp_positions = ode_interp_positions #- np.mean(ode_interp_positions, axis=(0, 1))[None,None,:]

    interp_positions = ode_interp_positions*distance_scale
    
    
    return times, interp_positions, masses


def generate_data(
    n_data: int, 
    basis_order: int, 
    n_masses:int, 
    sample_rate: int, 
    n_dimensions: int = 3, 
    detectors=["H1"], 
    window="none", 
    return_windowed_coeffs=True, 
    basis_type="chebyshev",
    data_type="newtonian-kepler") -> np.array:
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

    second = 1./(24*3600)
    n_samples = sample_rate
    times = np.linspace(0,1,sample_rate) #in days for ode
    solve_times = np.linspace(0,10000,sample_rate)

    G = 6.67430e-11  # gravitational constant (m^3 kg^-1 s^-2)
    c = 3e8

    # scale to 1 year
    second_scale = 86400
    # between 0 and au/100
    distance_scale = 1.4e11
    # between 0 and 100 solar masses
    mass_scale = 1.989e30

    position_scale = 2*distance_scale                             # in m
    velocity_scale = np.sqrt(2*G*mass_scale/distance_scale)*1e-1       # in m/s

    all_positions = np.zeros((n_data, n_masses, n_dimensions, n_samples))
    all_masses = np.zeros((n_data, n_masses))
    for i in range(n_data):
        masses, initial_positions, initial_velocities = resample_initial_conditions(
            times, 
            G, 
            n_masses, 
            n_dimensions, 
            position_scale, 
            mass_scale, 
            velocity_scale,
            data_type = data_type.split("-")[1]
            )

        t_times, positions, masses = solve_ode(
            times=solve_times,
            masses=masses, 
            initial_positions=initial_positions,
            initial_velocities=initial_velocities,
            n_samples=len(times))

        print(f"solved: {i}")
        all_positions[i] = np.transpose(positions, (1,2,0))
        all_masses[i] = masses

    # scale the positions and masses for use in the neural network
    all_positions = all_positions/distance_scale
    all_masses = all_masses/mass_scale

    return times, all_positions, all_masses, None

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
    