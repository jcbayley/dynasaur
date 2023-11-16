import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def newton_derivative(t, x_posvels, masses, factor=1, n_dimensions=3):
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
    massunit = 1.4e11*10
    G_ggravunits = G * ((86400**2)/((massunit)**3)) * 1.989e30
    c_ggravunits = c * ((86400**2)/(massunit))

    # reshape posvels to more useful shape
    x_posvels = x_posvels.reshape(n_masses, 2*n_dimensions)
    x_positions = x_posvels[:, 0:n_dimensions]
    x_vels = x_posvels[:, n_dimensions:2*n_dimensions]

    x_derivative = np.zeros((n_masses, 2*n_dimensions))
    for i, mass_1 in enumerate(masses):
        x_derivative[i][0:n_dimensions] = x_vels[i]
        for j, mass_2 in enumerate(masses):
            if i == j: continue
            separation = np.sqrt(np.sum((x_positions[i] - x_positions[j])**2))
            diff = x_positions[j] - x_positions[i]
            x_derivative[i][n_dimensions:2*n_dimensions] += G_ggravunits*mass_2*diff/separation**3
        
        """
        # get all other masses but this one
        other_positions = np.delete(x_positions, i)
        other_vels = np.delete(x_vels, i)
        # compute separations
        rs = np.sqrt(np.sum((other_positions - x_positions[j])**2))
        """
    #energy_loss_term = x_derivative[i][3:6] * factor

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
        print(np.shape(positions[object_ind]))
        interp = interp1d(old_times, positions[object_ind], kind="cubic")
        interp_dict[object_ind] = interp(new_times)
    return interp_dict

def solve_ode(n_masses=2, n_dimensions=3):

    second = 1./(24*3600)
    times = np.linspace(0,2*second,128)

    initial_conditions = get_initial_conditions(n_masses, n_dimensions)*0.0001
    initial_conditions[:,3:6] *= 4e3
    masses = get_masses(n_masses)*4e3

    ode = lambda t, x: newton_derivative(t, x, masses=masses, factor=1, n_dimensions=3)

    outputs = solve_ivp(
        ode,
        [min(times)-second, max(times) + second], 
        initial_conditions.flatten(), 
        tfirst=True,
        method="LSODA",
        rtol = 1e-5)
    
    if max(outputs.t) < max(times):
        outputs.t = np.append(outputs.t, max(times))
        outputs.y = np.append(outputs.y.T, outputs.y[:, -1:].T, axis=0).T

    positions = outputs.y[:, :]
    interp_positions = interpolate_positions(outputs.t, times, positions).T
    interp_positions = interp_positions.reshape(len(times), n_masses, 2*n_dimensions)[:,:,:3]


    return times, interp_positions, masses

if __name__ == "__main__":

    times, outputs, masses = solve_ode()

    print(outputs)

    fig, ax = plt.subplots()

    ax.plot(outputs[:, 0, 0], outputs[:, 0, 1], markersize = masses[0]*1e-2, marker="o")
    ax.plot(outputs[:, 1, 0], outputs[:, 1, 1], markersize = masses[1]*1e-2, marker="o")
    ax.plot(outputs[0, 0, 0], outputs[0, 0, 1], markersize = masses[0]*1e-2, marker="o", color="k")
    ax.plot(outputs[0, 1, 0], outputs[0, 1, 1], markersize = masses[1]*1e-2, marker="o", color="k")

    fig.savefig("./test_ode.png")