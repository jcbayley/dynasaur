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