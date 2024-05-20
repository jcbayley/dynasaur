import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
import timeit
from massdynamics.basis_functions import basis
from massdynamics.data_generation import orbits_functions, data_processing

def acceleration_1pn(m0, m1, r, v, rvect, vvect, G, c):

    prefact = 4*(G**2)/(5*(c**5)*(r**3))*m0*m1*(m1/(m0 + m1))

    fact1 = rvect*np.dot(rvect, vvect)*((34/3)*G*(m0+m1)/r + 6*v**2)

    fact2 = vvect*(-6*G*(m0+m1)/r - 2*v**2)
    #print(prefact, fact1, fact2)
    #print(c,G,m0,m1, r, v)
    return prefact * (fact1 + fact2)

def acceleration_c2(m1, m2, r12, v12, r1, r2, v1, v2, G, c):
    """taken from https://doi.org/10.12942/lrr-2014-2

    Args:
        m1: 
        m2: 
        r1 (_type_): _description_
        r2 (_type_): _description_
        v1 (_type_): _description_
        v2 (_type_): _description_
        G (_type_): _description_
        c (_type_): _description_

    Returns:
        _type_: _description_
    """
    #r12 = np.sum((r1 - r2)**2)
    n12 = (r1 - r2)/r12
    #v12 = v1 - v2

    fact1 = 5*(G**2)*m1*m2/(r12**3)

    fact2 = 4*(G**2)*(m2**2)/(r12**2)

    fact3 = G*m2/(r12**2)

    bracket1 = (3/2)*np.dot(n21*v2)**2 - np.dot(v1, v1) + 4*np.dot(v1,v2) - 2*np.dot(v2, v2)

    term1 = fact1 + fact2 + fact3*bracket2

    fact4 = G*m2/(r12**2)

    bracket2 = 2*np.dot(n12, v1) - 2*np.dot(n12,v2)

    term2 = fact4*bracket2

    return 1/c^2*(term1*n12 + term2*v12)

def acceleration_c4(m1, m2, r12, v12, r1, r2, v1, v2, G, c):
    """taken from https://doi.org/10.12942/lrr-2014-2

    Args:
        r1 (_type_): _description_
        r2 (_type_): _description_
        v1 (_type_): _description_
        v2 (_type_): _description_
        G (_type_): _description_
        c (_type_): _description_

    Returns:
        _type_: _description_
    """
    #r12 = np.sum((r1 - r2)**2)
    n12 = (r1 - r2)/r12
    #v12 = v1 - v2
    term1 = (
        (-57 * G ** 3 * m1 ** 2 * m2 ** 2 / (4 * r12 ** 4)) +
        (-69 * G ** 3 * m1 * m2 ** 2 / (2 * r12 ** 4)) +
        (-9 * G ** 3 * m2 ** 2 / (r12 ** 4)) +
        (G * m2 / (r12 ** 2)) * (
            (-15 / 8) * (np.dot(n12,v2) ** 4) +
            (3 / 2) * (np.dot(n12, v2) ** 2 * np.dot(v1, v1)) -
            6 * (np.dot(n1, v2) ** 2 * np.dot(v1, v2)) -
            2 * np.dot(v1, v2) ** 2 +
            (9 / 2) * (np.dot(n1, v2) ** 2 * np.dot(v2, v2)) +
            4 * np.dot(v1, v2) * np.dot(v2, v2) -
            2 * (np.dot(v2, v2) ** 4)
        ) +
        (G ** 2 * m1 * m2 / (r12 ** 3)) * (
            (39 / 2) * (np.dot(n12, v1) ** 2) -
            39 * (np.dot(n12 , v1) * np.dot(n12, v2)) +
            (17 / 2) * (np.dot(n12, v2) ** 2) -
            (15 / 4) * np.dot(v1, v1) -
            (5 / 2) * np.dot(v1 , v2) +
            (5 / 4) * np.dot(v2, v2)
        ) +
        (G ** 2 * m2 ** 2 / (r12 ** 3)) * (
            2 * (np.dot(n12, v1) ** 2) -
            4 * (np.dot(n12, v1) * np.dot(n12 , v2)) -
            6 * (np.dot(n12, v2) ** 2) -
            8 * np.dot(v1, v2) +
            4 * np.dot(v2, v2)
        )
    )

    term2 = (
        (G ** 2 * m2 ** 2 / (r12 ** 3)) * (
            -2 * (np.dot(n12, v1)) -
            2 * np.dot(n12, v2)
        ) +
        (G ** 2 * m1 * m2 / (r12 ** 3)) * (
            (-63 / 4) * np.dot(n12, v1) +
            (55 / 4) * np.dot(n12, v2)
        ) +
        (G * m2 / (r12 ** 2)) * (
            -6 * (np.dot(n12 * v1) * np.dot(n12, v2) ** 2) +
            (9 / 2) * (np.dot(n12, v2) ** 3) +
            (np.dot(n12, v2) * np.pdot(v1, v1)) -
            4 * (np.dot(n12, v1) * np.dot(v2, v2)) +
            4 * (np.dot(n12, v2) * np.dot(v2, v2)) +
            4 * (np.dot(n12, v1) * np.dot(v2, v2)) -
            5 * (np.dot(n12, v2) * np.dot(v2, v2))
        )
    )

    return (1/c**4) * (term1 * n12 + term2 * v12)

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
    #G = 6.67e-11
    #c = 3e8
    # compute G in gaussian gravitational units to prevent overflows
    # G * (s/day)/(m/Au) * Msun(kg)

    # reshape posvels to more useful shape
    x_posvels = x_posvels.reshape(n_masses, 2*n_dimensions)
    x_positions = x_posvels[:, 0:n_dimensions]
    x_vels = x_posvels[:, n_dimensions:2*n_dimensions]

    seps = np.zeros((n_masses, n_masses))
    decay = 0.9

    x_derivative = np.zeros((n_masses, 2*n_dimensions))
    for i, mass_1 in enumerate(masses):
        x_derivative[i][0:n_dimensions] = x_vels[i]
        for j, mass_2 in enumerate(masses):
            if i == j: continue
            separation = np.sqrt(np.sum((x_positions[i] - x_positions[j])**2) + EPS)
            separation_cubed = separation**3
            absvel = np.sqrt(np.sum((x_vels[i] - x_vels[j])**2))
            #seps[i,j] = separation_cubed
            diff = x_positions[j] - x_positions[i]
            veldiff = x_vels[i] - x_vels[j]
            acceleration = G*mass_2*diff/separation_cubed
            """
            acc_1pn = acceleration_1pn(
                mass_1, 
                mass_2, 
                separation, 
                absvel, 
                diff, 
                veldiff, 
                G, 
                c)
            """
            acc_c2 = acceleration_c2(mass_1, mass_2, separation, veldiff, x_positions[i], x_positions[j], x_vels[i], x_vels[j], G, c)
            acc_c4 = acceleration_c2(mass_1, mass_2, separation, veldiff, x_positions[i], x_positions[j], x_vels[i], x_vels[j], G, c)
            #print(acceleration, gw_acceleration)
            x_derivative[i][n_dimensions:2*n_dimensions] += acceleration + acc_c2 + acc_c4
         
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
    
def keplerian_to_cartesian(
        semi_major_axis, 
        eccentricity, 
        inclination, 
        long_ascending_node, 
        arg_periapsis, 
        true_anomaly, 
        mu=1.0):
    """_summary_

    Args:
        semi_major_axis (_type_): _description_
        eccentricity (_type_): _description_
        inclination (_type_): _description_
        long_ascending_node (_type_): _description_
        arg_periapsis (_type_): _description_
        mean_anomaly (_type_): _description_
        mu (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: _description_
    """

    # Convert angles to radians
    #i, Ω, ω, θ = np.radians(i), np.radians(Ω), np.radians(ω), np.radians(θ)

    # Compute orbital plane coordinates
    #r = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * np.cos(mean_anomoly))

    #eccentric_anomaly = mean_anomoly

    """
    true_anomaly = mean_anomaly 
    + (2*eccentricity - 1/4*eccentricity**3)*np.sin(mean_anomaly)
    + 5/4*eccentricity**2*np.sin(2*mean_anomaly) 
    + 13/12*eccentricity**3*np.sin(3*mean_anomaly)
    """
    eccentric_anomaly = np.arctan2(np.sqrt(1 - eccentricity**2) * np.sin(true_anomaly), eccentricity + np.cos(true_anomaly))

    #true_anomoly = 2*np.arctan2(np.sqrt(1+eccentricity)*np.sin(eccentric_anomaly/2), np.sqrt(1-eccentricity)*np.sin(eccentric_anomaly/2))

    r = semi_major_axis * (1 - eccentricity * np.cos(eccentric_anomaly))

    #x_prime = r * np.cos(true_anomoly)
    #y_prime = r * np.sin(true_anomoly)

    # Compute velocity in polar coordinates
    #v_r = np.sqrt(mu * (2 / r - 1 / semi_major_axis))
    #v_theta = v_r / r * np.sqrt(1 - eccentricity**2)
    v_r = semi_major_axis*np.sqrt(1 - eccentricity**2) * np.cos(eccentric_anomaly)
    v_theta = semi_major_axis * np.sin(eccentric_anomaly)

    # precompute some sines and cosines
    sin_inc = np.sin(inclination)
    cos_inc = np.cos(inclination)

    sin_periapsis = np.sin(arg_periapsis)
    cos_periapsis = np.cos(arg_periapsis)

    sin_ascending_node = np.sin(long_ascending_node)
    cos_ascending_node = np.cos(long_ascending_node)


    # Convert to inertial coordinates
    x = r * (cos_ascending_node * cos_periapsis 
                   - sin_ascending_node * sin_periapsis * cos_inc)
    y = r * (sin_ascending_node * cos_periapsis 
                   + cos_ascending_node * sin_periapsis * cos_inc)
    z = r * sin_periapsis * sin_inc

    # Convert polar velocities to Cartesian velocities
    v_x = v_r * (cos_ascending_node * cos_periapsis 
                 - sin_ascending_node * sin_periapsis * cos_inc) 
    - v_theta * (cos_ascending_node * sin_periapsis 
                 + sin_ascending_node * cos_periapsis * cos_inc)
    
    v_y = v_r * (sin_ascending_node * cos_periapsis 
                 + cos_ascending_node * sin_periapsis * cos_inc) 
    + v_theta * (cos_ascending_node * sin_periapsis 
                 - sin_ascending_node * sin_periapsis * cos_inc)
    
    v_z = v_r * sin_periapsis * sin_inc + v_theta * sin_periapsis * cos_inc

    return np.array([x, y, z]), np.array([v_x, v_y, v_z])

def kepler_apoapsis(semi_major_axis, eccentricity, theta, masses, G):
    """generate initial conditionss for a kepler orbit in 2d plane starting at apoapsis

    Args:
        semi_major_axis (_type_): _description_
        eccentricity (_type_): _description_
        theta (_type_): _description_
        masses (_type_): _description_
        G (_type_): _description_

    Returns:
        _type_: _description_
    """
    ra = semi_major_axis*(1+eccentricity)
    velocity = np.sqrt(G*np.sum(masses)/ra*(1-eccentricity))

    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0,0,0]])

    r0,v0 = np.zeros(3), np.zeros(3)

    r1 = np.array([ra,0,0])
    v1 = np.array([0,velocity,0])

    r0 = rotation_matrix.dot(masses[1]/np.sum(masses) * r0)
    r1 = rotation_matrix.dot(masses[0]/np.sum(masses) * r1)

    v0 = rotation_matrix.dot(masses[1]/np.sum(masses) * v0)
    v1 = rotation_matrix.dot(masses[0]/np.sum(masses) * v1)

    return np.concatenate([[r0],[r1]], axis=0), np.concatenate([[v0],[v1]], axis=0)

def kepler_apoapsis_binary(semi_major_axis, eccentricity, theta, masses, G):
    """generate initial conditionss for a kepler orbit in 2d plane starting at apoapsis

    Args:
        semi_major_axis (_type_): _description_
        eccentricity (_type_): _description_
        theta (_type_): _description_
        masses (_type_): _description_
        G (_type_): _description_

    Returns:
        _type_: _description_
    """
    ra = semi_major_axis*(1+eccentricity)
    velocity = np.sqrt(G*np.sum(masses)/ra*(1-eccentricity))

    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0,0,0]])

    r0 = np.array([-ra, 0, 0])
    v0 = np.array([0,-velocity, 0])

    r1 = np.array([ra,0,0])
    v1 = np.array([0,velocity,0])

    r0 = rotation_matrix.dot(masses[1]/np.sum(masses) * r0)
    r1 = rotation_matrix.dot(masses[0]/np.sum(masses) * r1)

    v0 = rotation_matrix.dot(masses[1]/np.sum(masses) * v0)
    v1 = rotation_matrix.dot(masses[0]/np.sum(masses) * v1)

    return np.concatenate([[r0],[r1]], axis=0), np.concatenate([[v0],[v1]], axis=0)

    
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
        masses = np.array([0.5,0.5])*mass_scale*1e2
        initial_positions, initial_velocities = get_initial_positions_velocities(n_masses, n_dimensions, position_scale, velocity_scale)
        
    elif data_type == "kepler_fixedperiod":
        M = 1e30
        duration = np.max(times) - np.min(times)
        n_samples = len(times)
        period = duration
        min_period = 2.*duration/n_samples

        M = mass_scale

        masses = np.array([M, np.random.uniform(1e-5*M, 1e-3*M)])

        semi_major_axes = (G*(np.sum(masses))/(4*np.pi**2) * period**2)**(1/3)
        eccentricities = np.random.uniform(0.0,0.9)
        #inclinations = np.array([0.0])
        #long_ascending_node = 0.0#np.random.uniform(0.0, 2*np.pi, size=1) # Longitude of the ascending node in degrees
        arg_periapsis = np.random.uniform(0.0, 2*np.pi) # Argument of periapsis in degrees
        #true_anomoly = np.random.uniform(0.0, 2*np.pi, size=1) # True anomaly in degrees

        initial_positions, initial_velocities = kepler_apoapsis(
            semi_major_axes, 
            eccentricities, 
            arg_periapsis, 
            masses, 
            G)
        """
        initial_positions, initial_velocities = keplerian_to_cartesian(
            semi_major_axes, 
            eccentricities, 
            inclinations, 
            long_ascending_node, 
            arg_periapsis, 
            true_anomoly, 
            mu=mu)

        # include central mass at 0,0 with no velocity
        initial_positions = np.concatenate([np.zeros(np.shape(initial_positions)), initial_positions], axis=1).T
        initial_velocities = np.concatenate([np.zeros(np.shape(initial_velocities)), initial_velocities], axis=1).T

        
        initial_positions, initial_velocities = generate_kepler_orbit(
            times[0], 
            semi_major_axes, 
            eccentricities, 
            inclinations, 
            G, 
            M, 
            periods=None)
    
        print("circ",np.sqrt(G*np.sum(masses)/semi_major_axes))
        print("comp",np.sqrt(np.sum(initial_velocities**2)))

        print(initial_velocities)
        
        # include central mass at 0,0 with no velocity
        initial_positions = np.concatenate([np.zeros(np.shape(initial_positions)), initial_positions], axis=0)
        initial_velocities = np.concatenate([np.zeros(np.shape(initial_velocities)), initial_velocities], axis=0)
        """
    elif data_type == "circular":
        M = 1e30
        duration = np.max(times) - np.min(times)
        n_samples = len(times)
        period = duration
        min_period = 2.*duration/n_samples
        max_period = duration/10

        M = mass_scale

        masses = 10*np.array([M, np.random.uniform(1e-3*M, 1e-2*M)])

        period = np.random.uniform(min_period, max_period)

        semi_major_axes = (G*(np.sum(masses))/(4*np.pi**2) * period**2)**(1/3)
        eccentricities = 0.0
        #inclinations = np.array([0.0])
        #long_ascending_node = 0.0#np.random.uniform(0.0, 2*np.pi, size=1) # Longitude of the ascending node in degrees
        arg_periapsis = np.random.uniform(0.0, 2*np.pi) # Argument of periapsis in degrees
        #true_anomoly = np.random.uniform(0.0, 2*np.pi, size=1) # True anomaly in degrees

        initial_positions, initial_velocities = kepler_apoapsis(
            semi_major_axes, 
            eccentricities, 
            arg_periapsis, 
            masses, 
            G)
        
    elif data_type == "circularbinary":
        M = 1e30
        duration = np.max(times) - np.min(times)
        n_samples = len(times)
        period = duration
        min_period = 2.*duration/n_samples
        max_period = min_period*2
        M = mass_scale

        masses = np.random.uniform(20, 30)*np.array([M, M])
        #masses = np.random.uniform(2, 1000)*np.array([M, M])

        period = np.random.uniform(min_period, max_period)

        semi_major_axes = (G*(np.sum(masses))/(4*np.pi**2) * period**2)**(1/3)
        eccentricities = 0.0
        #inclinations = np.array([0.0])
        #long_ascending_node = 0.0#np.random.uniform(0.0, 2*np.pi, size=1) # Longitude of the ascending node in degrees
        arg_periapsis = np.random.uniform(0.0, 2*np.pi) # Argument of periapsis in degrees
        #true_anomoly = np.random.uniform(0.0, 2*np.pi, size=1) # True anomaly in degrees

        initial_positions, initial_velocities = kepler_apoapsis_binary(
            semi_major_axes, 
            eccentricities, 
            arg_periapsis, 
            masses, 
            G)

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


    """
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
        
        print(oenergy)
        print(initial_positions)
        print(initial_velocities)
        sys.exit()
        
        n_resampled += 1
    #print("N_resampled: ", n_resampled)
    """
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

    ode_times = times/second_scale
    G_ggravunits = G * ((second_scale**2)/((distance_scale)**3)) * mass_scale
    c_ggravunits = c * ((second_scale**2)/(distance_scale))

    n_masses, n_dimensions = np.shape(initial_positions)

    #G_ggravunits = G * ((second_scale**2)/((distance_scale)**3)) * mass_scale
    #c_ggravunits = c * ((second_scale**2)/(distance_scale))

    initial_positions = data_processing.subtract_center_of_mass(initial_positions[:,:,np.newaxis], masses)[:,:,0]

    ode_initial_velocities = initial_velocities*second_scale/distance_scale # now in AU/day
    ode_initial_positions = initial_positions/distance_scale                # now in AU
    ode_masses = masses/mass_scale


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
        mass_scale=mass_scale,
        G=G_ggravunits,
        c=c_ggravunits)
    
    #too_close_event.terminal = True

    outputs = solve_ivp(
        ode,
        t_span=[min(ode_times), max(ode_times)], 
        y0=initial_conditions.flatten(), 
        tfirst=True,
        method="RK45",
        rtol=1e-6,
        atol=1e-6)
    
    """
    if max(outputs.t) < max(times):
        outputs.t = np.append(outputs.t, max(times)+0.1)
        outputs.y = np.append(outputs.y.T, outputs.y[:, -1:].T, axis=0).T
    """
    # y is shape (nvals, ntimes, )

    print(outputs.y)

    positions = outputs.y.reshape(n_masses, 2*n_dimensions, len(outputs.t))[:,:3] # get positions only
    positions = positions.reshape(n_masses*n_dimensions, len(outputs.t))

    ode_interp_positions = interpolate_positions(outputs.t, ode_times, positions).T # ntimes, nvals
    ode_interp_positions = ode_interp_positions.reshape(len(ode_times), n_masses, n_dimensions)
    ode_interp_positions = ode_interp_positions #- np.mean(ode_interp_positions, axis=(0, 1))[None,None,:]
    
    #scaled_interp_positions = ode_interp_positions*distance_scale/position_scale
    scaled_interp_positions = ode_interp_positions*distance_scale                # now in AU
    scaled_masses = masses*mass_scale

    times = outputs.t
    scaled_interp_positions = positions*distance_scale
    
    return times, scaled_interp_positions, scaled_masses


def generate_data(
    n_data: int, 
    basis_order: int, 
    n_masses:int, 
    sample_rate: int, 
    n_dimensions: int = 3, 
    detectors=["H1"], 
    window="none", 
    window_acceleration=True, 
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
    solve_times = np.linspace(0,2,sample_rate)

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
            solve_times, 
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
    