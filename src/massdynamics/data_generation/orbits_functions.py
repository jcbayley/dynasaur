import numpy as np

def compute_orbital_energy(position, velocity, G, M, m):
    """compute the orvital energy 

    Args:
        position (_type_): (N, ..., n_dimensions)
        velocity (_type_): (N, ...., n_dimensions)
        G (_type_): gravitational constant
        M (_type_): central Mass
        m (_type_): orbiting mass

    Returns:
        _type_: total energy or orvit
    """
    # Calculate the distance from the central body
    r = np.linalg.norm(position, axis=-1)

    # Calculate kinetic energy
    kinetic_energy = 0.5 * m * np.linalg.norm(velocity, axis=-1)**2

    # Calculate potential energy
    potential_energy = - G * M * m / r

    # Total energy is the sum of kinetic and potential energy
    total_energy = kinetic_energy + potential_energy

    return total_energy

def orbital_energy(M1, M2, r1, r2, v1, v2, G):
    """
    Compute the orbital energy of two orbiting objects.

    Parameters:
    - M1, M2: masses of the two objects
    - r1, r2: positions of the two objects as numpy arrays
    - v1, v2: velocities of the two objects as numpy arrays

    Returns:
    - Orbital energy (float)
    """
    G = 6.67430e-11  # Gravitational constant in m^3 kg^(-1) s^(-2)

    # Calculate the relative position and velocity vectors
    relative_position = r1 - r2
    relative_velocity = v1 - v2

    # Calculate the reduced mass
    mu = (M1 * M2) / (M1 + M2)

    # Calculate the orbital energy
    kinetic_energy = 0.5 * mu * np.dot(relative_velocity, relative_velocity)
    potential_energy = - (G * M1 * M2) / np.linalg.norm(relative_position)

    energy = kinetic_energy + potential_energy
    return energy

def compute_angular_momentum(position, velocity, m):
    """compute total angular momentum as a function of time

    Args:
        position (_type_): (N, ..., n_dimensions)
        velocity (_type_): (N, ..., n_dimensions)
        m (_type_): object mass

    Returns:
        np.array: total_angular_momentum (N, ...)
        np.array: angular_momentum_vector (N, ..., n_dimensions)
    """
    # Cross product of position and linear momentum
    angular_momentum_vector = np.cross(position, m * velocity, axis=-1)

    # Magnitude of the angular momentum
    angular_momentum_magnitude = np.linalg.norm(angular_momentum_vector, axis=-1)

    return angular_momentum_magnitude, angular_momentum_vector