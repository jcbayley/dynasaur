import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time
import timeit
from massdynamics.basis_functions import basis
from massdynamics.data_generation import orbits_functions, data_processing
from massdynamics.data_generation.models.newtonian_eom import newton_derivative


def interpolate_positions(old_times, new_times, positions):
    """Interpolate between points over dimension 1 """
    interp_dict = np.zeros((positions.shape[0], len(new_times)))
    for object_ind in range(positions.shape[0]):
        #print(np.shape(positions[object_ind]))
        interp = interp1d(old_times, positions[object_ind], kind="cubic")
        interp_dict[object_ind] = interp(new_times)
    return interp_dict


def get_prior(times, prior_args):

    masses = np.random.uniform(prior_args["mass_min"], prior_args["mass_max"], size=2)

    m1 = masses[0]
    m2 = masses[1]
    if m2 > m1:
        masses = np.array([m2, m1])

    duration = times.max() - times.min()

    period = np.random.uniform(duration/prior_args["cycles_max"], duration/prior_args["cycles_min"])
    initial_phase = np.random.uniform(prior_args["initial_phase_min"], prior_args["initial_phase_max"])
    if prior_args["inclination_min"] == "faceoff":
        inclination = np.random.choice([0,np.pi])
    else:
        inclination = np.random.uniform(prior_args["inclination_min"], prior_args["inclination_max"])

    long_ascending_node = np.random.uniform(prior_args["long_ascending_node_min"], prior_args["long_ascending_node_max"])

    return masses, period, initial_phase, inclination, long_ascending_node


def get_positions(times, masses, period, phase, inclination, long_ascending_node, G=6.67e-11):
    """_summary_

    Args:
        times (_type_): _description_
        masses (_type_): _description_
        period (_type_): _description_
        phase (_type_): _description_
        inclination (_type_): _description_
        G (_type_, optional): _description_. Defaults to 6.67e-11.
    """

    omega = 2*np.pi/period
    M = np.sum(masses)
    r = (period**2/(4*np.pi*np.pi) * G * M)**(1/3)

    cwt = np.cos(omega*times + phase)
    swt = np.sin(omega*times + phase)
    comega = np.cos(omega)
    somega = np.sin(omega)
    cinc = np.cos(inclination)
    sinc = np.sin(inclination)

    xp = r*(comega*cwt - somega*cinc*swt)
    yp = r*(somega*cwt + comega*cinc*swt)
    zp = r*(sinc*swt)

    r12 = np.array([xp, yp, zp])

    r0 = masses[1]/M * r12
    r1 = -masses[0]/M * r12

    return np.array([r0, r1])

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
    data_type="newtonian-kepler",
    prior_args = {}) -> np.array:
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

    newtoniandecay = True if data_type.split("-")[0] in ["newtoniandecay", "newtonian_decay"] else False

    ntimeseries = [0, 1, 3, 6, 10]

    strain_timeseries = np.zeros((n_data, len(detectors), sample_rate))

    prior_args.setdefault("sample_rate", sample_rate)
    prior_args.setdefault("n_samples", sample_rate)
    prior_args.setdefault("duration", 1)
    prior_args.setdefault("inclination_min", "faceoff")
    prior_args.setdefault("masses_min", 0.5)
    prior_args.setdefault("masses_max", 1)
    prior_args.setdefault("initial_phase_min", 0)
    prior_args.setdefault("initial_phase_max", 2*np.pi)
    prior_args.setdefault("cycles_min", 1)
    prior_args.setdefault("cycles_max", prior_args["sample_rate"]/4)
    prior_args.setdefault("long_ascending_node_min", 0.0)
    prior_args.setdefault("long_ascending_node_max", 0.0)

    times = np.linspace(0,prior_args["duration"],prior_args["n_samples"]) 

    all_positions = np.zeros((n_data, n_masses, n_dimensions, prior_args["n_samples"]))
    all_masses = np.zeros((n_data, n_masses))
    for i in range(n_data):

        masses, period, initial_phase, inclination, long_ascending_node = get_prior(times, prior_args)

        positions = get_positions(times, masses, period, initial_phase, inclination, long_ascending_node, G=1)

        #print(f"solved: {i}")
        all_positions[i] = positions
        all_masses[i] = masses

    # scale the positions and masses for use in the neural network
    all_positions = all_positions
    all_masses = all_masses

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
    