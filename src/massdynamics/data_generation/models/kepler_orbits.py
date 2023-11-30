import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
import timeit
from massdynamics.basis_functions import basis

def newton_derivative_vect(
    t, 
    x_posvels, 
    masses, 
    factor=1, 
    n_dimensions=3, 
    EPS=1e-11,
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
    #G_ggravunits = G * ((second_scale**2)/((distance_scale)**3)) * mass_scale
    #c_ggravunits = c * ((second_scale**2)/(distance_scale))

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
    x_derivative[:, 3:6] = np.einsum("ijk, ik, k->ij", G*diff_xyz, inv_r_cubed, masses)
    #x_derivative[:, 3:6] = np.einsum("ijk,k->ij", div_r_cubed, masses)

    return x_derivative.flatten()

def compute_angular_momentum(r, v):
    return np.cross(r, v)

def compute_eccentricity(r, v, m, G):
    r = np.sqrt(np.sum(r**2))
    v = np.sqrt(np.sum(v**2))
    return 1./(2/r - v**2/(G*M))

def compute_eccentricity2(r, v, m, G):
    L = compute_angular_momentum(r, v)
    e = 1./(G*M) * np.cross(v, L)
    return e

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
        v_t = np.sqrt(2 * G * M / r - G * M / a)

        # Convert polar velocities to Cartesian velocities
        vx = v_r * np.cos(theta) - v_t * np.sin(theta)
        vy = v_r * np.sin(theta) + v_t * np.cos(theta)
        vz = np.zeros(np.shape(vx))

        # Append positions and velocities
        positions.append(np.array([x, y, z]))
        velocities.append(np.array([vx, vy, vz]))

    return positions, velocities

def unnormalise_initial_conditions(initial_conditions, second_scale, distance_scale, mass_scale):
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

def normalise_initial_conditions(initial_conditions, second_scale, distance_scale, mass_scale):
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
    return masses/masssum[:, None]

def interpolate_positions(old_times, new_times, positions):
    """Interpolate between points """
    interp_dict = np.zeros((positions.shape[0], len(new_times)))
    for object_ind in range(positions.shape[0]):
        interp = interp1d(old_times, positions[object_ind], kind="cubic")
        interp_dict[object_ind] = interp(new_times)
    return interp_dict


def generate_data(
    n_samples,
    detectors = ["H1", "L1", "V1"],
    basis_order = 16,
    n_masses = 1,
    basis_type = "fourier",
    n_dimensions = 3,
    sample_rate=16,
    fixed_period=False):

    n_detectors = len(detectors)
    # scale to 1 year
    second_scale = 86400
    # between 0 and au/100
    distance_scale = 1.4e11
    # between 0 and 100 solar masses
    mass_scale = 1.989e30

    G = 6.67430e-11  # gravitational constant (m^3 kg^-1 s^-2)
    c = 3e8
    M = 1*mass_scale     # mass of the sun (kg)

    G_scaled = G * ((second_scale**2)/((distance_scale)**3)) * mass_scale
    c_scaled = c * ((second_scale**2)/(distance_scale))

    """
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
    """
    masses = get_masses(n_samples, n_masses)*mass_scale/1e5

    # priors currently fixed
    #times = np.linspace(0,5e7, sample_rate)
    duration = 5e7
    times = np.arange(0,duration,duration/sample_rate)
    output_times = times/duration
    n_times = len(times)

    fixed_period = True
    period = duration
    min_period = 2.*duration/sample_rate

    # define the semimajor axis with a fixed period for each of the masses
    if fixed_period:
        semi_major_axis = (G*(M + masses[:,0])/(4*np.pi**2) * period**2)**(1/3)
    else:
        max_semi_major_axis = (G*M/(4*np.pi**2) * min_period**2)**(1/3)
        semi_major_axes = np.random.uniform(0.3,1,size=n_samples)*distance_scale
    eccentricities = np.random.uniform(0.0,0.9,size=n_samples)
    inclinations = np.random.uniform(0.0,2*np.pi,size=n_samples)

    positions, velocities = generate_kepler_orbit(
        times, 
        semi_major_axes, 
        eccentricities, 
        inclinations, 
        G, 
        M)

    positions = np.array(positions)/distance_scale
    velocities = np.array(velocities)*second_scale/distance_scale

    # currently shape (n_samples, n_dimensions, n_times)
    initial_positions = positions#np.vstack([positions, velocities])
    # now shape (n_samples, n_masses, n_dimensions, n_times)
    initial_positions = np.expand_dims(initial_positions, 1)


    return output_times, initial_positions, masses, None

if __name__ == "__main__":

    n_orbits = 5
    times, positions, masses,  = generate_data(
        n_orbits,
        detectors = ["H1", "L1", "V1"],
        basis_order = 128,
        basis_type = "fourier",
        n_dimensions = 3)

    #print(outputs)
    print(np.shape(positions))
    fig, ax = plt.subplots()
    ax.plot(0,0,marker="o", ms=5)
    for i in range(n_orbits):
        ax.plot(positions[i, 0, 0, :], positions[i, 0, 1, :], markersize = 2, marker="o")
    fig.savefig("./test_orbits.png")


    