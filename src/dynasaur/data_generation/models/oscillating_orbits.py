import numpy as np



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
    data_type="oscillatex",
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

        f = np.random.uniform(prior_args["cycles_min"], prior_args["cycles_max"])
        t1_positions = np.array([np.sin(2*np.pi*f*times), np.sin(2*np.pi*f*times), np.zeros(len(times))])
        t2_positions = np.array([-np.sin(2*np.pi*f*times), -np.sin(2*np.pi*f*times), np.zeros(len(times))])
        t_positions = np.array([t1_positions, t2_positions])
        t_masses = np.random.uniform(0.5,0.5, size=2)
        all_positions[i] = t_positions
        all_masses[i] = t_masses

    return times, all_positions, all_masses, None