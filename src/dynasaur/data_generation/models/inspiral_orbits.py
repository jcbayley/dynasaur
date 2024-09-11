import numpy as np

def generate_m1m2_pos(times, m1, m2, tc, orientation="xy"):
    """Generate the positions of the inspiral to first order

    Args:
        times (array): times at which to evaluate the signal
        m1 (float): mass of object 1
        m2 (float): mass of object 2
        tc (float): time of coalescence

    Returns:
        array: array of the normalised masses
        array: array of the positions of each mass
    """
    #basis_order = 30
    G = 6.6e-11
    c = 3e8
    M_sun = 1.9e30
    r_s = 2*G*M_sun/c**2
    # chirp mass
    Mc = ((m1*m2)**(3./5.) )/(m1 + m2)**(1./5.)
    # orbital frequency
    f = 0.5*( (8*np.pi)**(8./3.)/5. * (G*Mc*M_sun/c**3)**(5./3.) *(tc-times) )**(-3./8.) 
    # orbital separation
    r = (( G * M_sun*(m1 + m2)/(2*np.pi*f)**2 )**(1./3.))/r_s 

    sm1 = m1/(m1+m2)
    sm2 = m2/(m1+m2)
    r1 = r*sm2
    r2 = r*sm1
    theta = 2*np.pi*f*times

    if orientation == "xy":
        m1pos = np.vstack([-r1*np.cos(theta), -r1*np.sin(theta), np.zeros(np.shape(theta))])
        m2pos = np.vstack([r2*np.cos(theta), r2*np.sin(theta), np.zeros(np.shape(theta))])
    elif orientation == "xz":
        m1pos = np.vstack([-r1*np.cos(theta), np.zeros(np.shape(theta)), -r1*np.sin(theta)])
        m2pos = np.vstack([r2*np.cos(theta), np.zeros(np.shape(theta)), r2*np.sin(theta)])
    elif orientation == "yz":
        m1pos = np.vstack([np.zeros(np.shape(theta)), -r1*np.cos(theta), -r1*np.sin(theta)])
        m2pos = np.vstack([np.zeros(np.shape(theta)), r2*np.cos(theta), r2*np.sin(theta)])
    else:
        raise Exception(f"No orientation {orientation}")


    positions = np.array([m1pos,m2pos])

    positions = positions/np.max(positions)

    masses = np.array([m1,m2])
    norm_masses = masses/np.sum(masses)

    return times, positions, norm_masses, None


def generate_data(
    n_data: int, 
    basis_order: int, 
    n_masses:int, 
    sample_rate: int, 
    n_dimensions: int = 3, 
    detectors=["H1"], 
    window_strain="none", 
    window_acceleration=True, 
    basis_type="fourier",
    data_type="inspiral",
    prior_args = {}) -> np.array:

    prior_args.setdefault("mass_min", 30)
    prior_args.setdefault("mass_max", 30)
    prior_args.setdefault("n_samples", sample_rate)
    m1, m2 = np.random.uniform(prior_args["mass_min"], prior_args["mass_max"]), np.random.uniform(prior_args["mass_min"], prior_args["mass_max"])

    tc = 1.1 # merger time just after time window as only looking at inspiral
    times = np.linspace(0,1,prior_args["n_samples"]) #in days for ode
    all_positions = np.zeros((n_data, n_masses, n_dimensions, prior_args["n_samples"]))
    all_masses = np.zeros((n_data, n_masses))
    for i in range(n_data):

        t_times, t_positions, t_masses, _ = generate_m1m2_pos(times, m1, m2, 1.1, orientation="xy")

        all_positions[i] = t_positions
        all_masses[i] = t_masses

    return times, all_positions, all_masses, None