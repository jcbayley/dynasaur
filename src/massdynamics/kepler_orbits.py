import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
import timeit


def get_initial_conditions(n_samples, n_masses, n_dimensions):
    """get positions and velocities
    of 2d ellipse

    Args:
        n_samples (int): How many so generate
        n_masses (_type_): _description_
        n_dimensions (_type_): _description_

    Returns:
        _type_: _description_
    """

    initial_position = np.random.uniform(-1,1,size=(n_samples, n_masses, n_dimensions))
    eccentricity = np.random.uniform(0,0.9, size=(n_samples, n_masses))

    # from position and eccentricity compute allowed semimajor axes
    semimajor_axis_min = initial_position/(1 + eccentricity[:,:,None])
    semimajor_axis_max = initial_position/(1 - eccentricity[:,:,None])
    semimajor_axis = np.random.uniform(semimajor_axis_min, semimajor_axis_max)

    print(np.shape(semimajor_axis))

    # compute velocity from vis-viva eqn
    v = np.sqrt(G*mass(2/initial_position - 1./semimajor_axis))


    return np.random.uniform(-1,1,size=(n_samples, n_masses, 2*n_dimensions))

def generate_kepler_orbit(num_orbits, semi_major_axes, eccentricities, inclinations, G, M):
    
    # Generate initial conditions for Keplerian orbits
    positions = []
    velocities = []

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
        t = 0#np.linspace(0, period, 1000)

        # Calculate mean anomaly
        mean_anomaly = 2 * np.pi * t / period

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
        v_t = np.sqrt(2 * G * M / r - G * M / a)

        # Convert polar velocities to Cartesian velocities
        vx = v_r * np.cos(theta) - v_t * np.sin(theta)
        vy = v_r * np.sin(theta) + v_t * np.cos(theta)
        vz = np.zeros(np.shape(vx))

        # Append positions and velocities
        positions.append(np.array([x, y, z]))
        velocities.append(np.array([vx, vy, vz]))

    return positions, velocities

def scale_initial_conditions(initial_conditions, second_scale, distance_scale, mass_scale):
    """scale initial conditions from 0-1 to those with units

    Args:
        initial_conditions (_type_): (Nsamps, 6) positions and velocities
        second_scale (_type_): _description_
        distance_scale (_type_): _description_
        mass_scale (_type_): _description_

    Returns:
        _type_: _description_
    """
    initial_conditions[:,0:3] = initial_conditions[:,0:3] * distance_scale
    initial_conditions[:,3:6] = initial_conditions[:,3:6] * distance_scale / second_scale

    return initial_conditions

def unscale_initial_conditions(initial_conditions, second_scale, distance_scale, mass_scale):
    """rescale init conditions with units back to 0-1

    Args:
        initial_conditions (_type_): (Nsamps, 6)
        second_scale (_type_): _description_
        distance_scale (_type_): _description_
        mass_scale (_type_): _description_

    Returns:
        _type_: _description_
    """
    inv_distance_scale = 1./distance_scale
    initial_conditions[:,0:3] = initial_conditions[:,0:3] * inv_distance_scale
    initial_conditions[:,3:6] = initial_conditions[:,3:6] * inv_distance_scale * second_scale

    return initial_conditions


def get_masses(n_samples, n_masses):

    masses = np.random.uniform(0,1,size=(n_samples, n_masses))
    masssum = np.sum(masses, axis=1)
    return masses/masssum[None, :]

def interpolate_positions(old_times, new_times, positions):
    """Interpolate between points """
    interp_dict = np.zeros((positions.shape[0], len(new_times)))
    for object_ind in range(positions.shape[0]):
        interp = interp1d(old_times, positions[object_ind], kind="cubic")
        interp_dict[object_ind] = interp(new_times)
    return interp_dict


def solve_ode(
    n_masses=1, 
    central_mass=10,
    n_dimensions=3, 
    n_samples=128,
    initial_conditions = None
    ):


    second = 1./(24*3600)
    times = np.linspace(0,2*second,n_samples)

    normed_initial_conditions = get_initial_conditions(
        n_masses, 
        n_dimensions)

    units_initial_conditions = scale_initial_conditions(
        normed_initial_conditions,
        second_scale,
        distance_scale,
        mass_scale
    )
    #initial_conditions[:,3:6] *= 1e4
    masses = get_masses(n_masses)

    ode = lambda t, x: newton_derivative_vect(
        t, 
        x, 
        masses=masses, 
        factor=1, 
        G = G_scaled,
        c = c_scaled)

    outputs = solve_ivp(
        ode,
        [min(times) - second, max(times) + second], 
        initial_conditions.flatten(), 
        tfirst=True,
        method="RK4",
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

def generate_data(n_samples):

    G = 6.67430e-11  # gravitational constant (m^3 kg^-1 s^-2)
    M = 1.989e30     # mass of the sun (kg)

    # scale to 1 year
    second_scale = 86400
    # between 0 and au/100
    distance_scale = 1.4e11
    # between 0 and 100 solar masses
    mass_scale = 1.989e30

    G_scaled = G * ((second_scale**2)/((distance_scale)**3)) * mass_scale
    c_scaled = c * ((second_scale**2)/(distance_scale))

    normed_initial_conditions = get_initial_conditions(
        n_samples,
        n_masses, 
        n_dimensions)

    units_initial_conditions = scale_initial_conditions(
        normed_initial_conditions,
        second_scale,
        distance_scale,
        mass_scale
    )

    masses = get_masses(n_samples, n_masses)

    num_orbits = 20
    semi_major_axes = np.random.uniform(0.1e11,10e11, size=num_orbits)
    eccentricities = np.random.uniform(0.0,0.9, size=num_orbits)
    inclinations = np.linspace(0.0,np.pi,num_orbits)
    times = np.linspace(0,52e7, 64)

    # generate lots of orbits over time interval
    positions, velocities = generate_kepler_orbit(
        times,
        num_orbits, 
        semi_major_axes, 
        eccentricities, 
        inclinations)

    scaled_positions = positions/distance_scale

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
    