import numpy as np
import lal
import lalpulsar 
import matplotlib.pyplot as plt
import scipy.signal as signal
import argparse
import h5py
import os
import torch
from massdynamics import window_functions
from massdynamics.data_generation import (
    compute_waveform,
    data_processing,
)
from massdynamics.data_generation.models import (
    random_orbits,
    newtonian_orbits,
    newtonian_orbits_decay,
    kepler_orbits,
    inspiral_orbits,
    oscillating_orbits
)
from massdynamics.basis_functions import basis


def generate_data(
    n_data: int, 
    basis_order: int, 
    n_masses:int, 
    sample_rate: int, 
    n_dimensions: int = 1, 
    detectors=["H1"], 
    window="none", 
    window_acceleration=True, 
    basis_type="chebyshev",
    data_type = "random",
    fourier_weight=0.0,
    coordinate_type="cartesian",
    noise_variance = 0.0,
    prior_args={}):

    if data_type.split("-")[0] == "random":
        times, positions, masses, position_coeffs = random_orbits.generate_data(
                n_data, 
                basis_order, 
                n_masses, 
                sample_rate, 
                n_dimensions, 
                detectors=detectors, 
                window=window, 
                window_acceleration=window_acceleration, 
                basis_type=basis_type,
                fourier_weight=fourier_weight,
                data_type=data_type,
                prior_args=prior_args)
    elif data_type.split("-")[0] in ["newtonian", "newtonian_decay", "newtoniandecay"]:
        times, positions, masses, position_coeffs = newtonian_orbits.generate_data(
                n_data, 
                basis_order, 
                n_masses, 
                sample_rate, 
                n_dimensions, 
                detectors=detectors, 
                window=window, 
                window_acceleration=window_acceleration, 
                basis_type=basis_type,
                data_type=data_type,
                prior_args=prior_args)
    elif data_type.split("-")[0] in ["inspiral"]:
        times, positions, masses, position_coeffs = inspiral_orbits.generate_data(
                n_data, 
                basis_order, 
                n_masses, 
                sample_rate, 
                n_dimensions, 
                detectors=detectors, 
                window=window, 
                window_acceleration=window_acceleration, 
                basis_type=basis_type,
                data_type=data_type,
                prior_args=prior_args)
    elif data_type.split("-")[0] in ["oscillatex"]:
        times, positions, masses, position_coeffs = oscillating_orbits.generate_data(
                n_data, 
                basis_order, 
                n_masses, 
                sample_rate, 
                n_dimensions, 
                detectors=detectors, 
                window=window, 
                window_acceleration=window_acceleration, 
                basis_type=basis_type,
                data_type=data_type,
                prior_args=prior_args)
    elif data_type == "kepler":
        times, positions, masses, position_coeffs = kepler_orbits.generate_data(
                n_data,
                detectors = detectors,
                n_masses=n_masses,
                basis_order = basis_order,
                basis_type = basis_type,
                n_dimensions = n_dimensions,
                sample_rate = sample_rate,
                prior_args=prior_args)
    else:
        raise Exception(f"No data with name {data_type.split('-')[0]}")
    
    if basis_type == "fourier":
        dtype = complex
    else:
        dtype = np.float64

    output_coeffs_mass = np.zeros((n_data, basis_order*n_masses*n_dimensions + n_masses))
    all_time_dynamics = np.zeros((n_data, n_masses, n_dimensions, len(times)))
    strain_timeseries = np.zeros((n_data, len(detectors), len(times)))
    if basis_type == "fourier":
        all_basis_dynamics = np.zeros((n_data, n_masses, n_dimensions, int(0.5*basis_order + 1)), dtype=dtype)
        all_basis_dynamics_coord = np.zeros((n_data, n_masses, n_dimensions, int(0.5*basis_order + 1)), dtype=dtype)

    else:
        all_basis_dynamics = np.zeros((n_data, n_masses, n_dimensions, basis_order), dtype=dtype)
        all_basis_dynamics_coord = np.zeros((n_data, n_masses, n_dimensions, basis_order), dtype=dtype)


    if positions is None:
        no_positions = True
        positions = np.zeros((n_data, n_masses, n_dimensions, len(times)))
    else:
        no_positions = False

    if window_acceleration not in [False, None, "none"]:
        window_coeffs = window_functions.get_window_coeffs(times, window_acceleration, order=basis_order, basis_type=basis_type, alpha=0.5)


    all_masses = np.zeros((n_data, n_masses))

    for data_index in range(n_data):

        all_masses[data_index] = masses[data_index]

        #temp_output_coeffs = np.zeros((n_masses, n_dimensions, acc_basis_order))
        t_basis_order = int(0.5*basis_order + 1) if basis_type == "fourier" else basis_order-1

        if position_coeffs is not None:
            positions[data_index] = basis[basis_type]["val"](
                times,
                position_coeffs[data_index]
            )
            #if n_masses > 1:
            #    positions[data_index] = data_processing.subtract_center_of_mass(positions[data_index], masses[data_index])

        # move to center of mass frane 
        if n_masses > 1:
            positions[data_index] = data_processing.subtract_center_of_mass(positions[data_index], masses[data_index])
            

        if coordinate_type == "spherical":
            positions_coord = data_processing.cartesian_to_spherical(positions[data_index])
        elif coordinate_type == "cartesian":
            positions_coord = positions[data_index]
        else:
            raise Exception(f"Coordinate type {coordinate_type} is not supported")

        for mass_index in range(n_masses):
            temp_coeffs = basis[basis_type]["fit"](
                times,
                positions[data_index,mass_index, :, :],
                t_basis_order
                )
            
            if window_acceleration not in [False, None, "none"]:
                #print(np.shape(temp_coeffs), np.shape(window_coeffs))
                temp_coeffs  = window_functions.window_coeffs(times, temp_coeffs.T, window_coeffs, basis_type=basis_type).T
            else:
                temp_coeffs = temp_coeffs
            #print(np.max(positions[data_index, mass_index]),np.max(temp_coeffs))
            # if windowing applied create coeffs which are windowed else just use the random coeffs
            #print(np.shape(positions), np.shape(temp_coeffs))
            all_basis_dynamics[data_index, mass_index] = temp_coeffs

            if coordinate_type != "cartesian":
    
                temp_coeffs2 = basis[basis_type]["fit"](
                    times,
                    positions_coord[mass_index, :, :][np.newaxis,:],
                    t_basis_order
                    )
                
                all_basis_dynamics_coord[data_index, mass_index] = temp_coeffs2

        
        strain_timeseries[data_index], energy = compute_waveform.get_waveform(
            times, 
            masses[data_index], 
            all_basis_dynamics[data_index], 
            detectors, 
            basis_type=basis_type,
            compute_energy=False)

        all_time_dynamics[data_index] = compute_waveform.get_time_dynamics(
            all_basis_dynamics[data_index], 
            times, 
            basis_type=basis_type)

        #print(np.max(all_time_dynamics))


    feature_shape = np.prod(np.shape(all_basis_dynamics)[1:]) + len(all_masses[0])
    #samples_shape, feature_shape = np.shape(output_coeffs_mass)

    if noise_variance != 0 and noise_variance != False:
        strain_timeseries = strain_timeseries + np.random.normal(0, noise_variance, size=np.shape(strain_timeseries))

    return times, all_basis_dynamics, all_masses, strain_timeseries, feature_shape, all_time_dynamics, all_basis_dynamics


def get_data_path(
    basis_order: int = 8,
    basis_type: str = "chebyshev",
    n_masses: int = 2,
    sample_rate: int = 128,
    n_dimensions: int = 3,
    detectors: list = ["H1", "L1", "V1"],
    window: str = "none",
    window_acceleration = False,
    data_type: str = "random"
    ):

    path = os.path.join(f"data_{data_type}_{basis_type}{basis_order}_mass{n_masses}_ndim{n_dimensions}_fs{sample_rate}_det{len(detectors)}_win{window}")

    return path

def save_data(
    data_dir: str, 
    data_split: int = 100000,
    n_examples: int = 10000,
    basis_order: int = 8,
    n_masses: int = 2,
    sample_rate: int = 128,
    n_dimensions: int = 3,
    detectors: list = ["H1", "L1", "V1"],
    window: str = "none",
    window_acceleration = False,
    basis_type: str = "chebyshev",
    data_type: str = "random",
    start_index: int = 0,
    fourier_weight:float=0.0,
    noise_variance=0.0
    ):


    data_path = get_data_path(
        basis_order = basis_order,
        basis_type = basis_type,
        n_masses = n_masses,
        sample_rate = sample_rate,
        n_dimensions = n_dimensions,
        detectors = detectors,
        window = window,
        window_acceleration = False,
        data_type = data_type)

    data_dir = os.path.join(data_dir, data_path)

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)

    times, labels, strain, cshape, positions, all_d = generate_data(
        2, 
        basis_order, 
        n_masses, 
        sample_rate, 
        n_dimensions=n_dimensions, 
        detectors=detectors, 
        window=window, 
        window_acceleration=window_acceleration,
        basis_type=basis_type,
        data_type=data_type,
        fourier_weight=fourier_weight,
        noise_variance=noise_variance)


    if n_examples < data_split:
        nsplits = 1
        data_split = n_examples
    else:
        nsplits = np.round(n_examples/data_split).astype(int)

    with h5py.File(os.path.join(data_dir, "metadata.hdf5"), "w") as f:
        f.create_dataset("times", data=np.array(times))
        f.create_dataset("poly_order", data=np.array(cshape))

    for split_ind in range(nsplits):

        times, t_labels, t_strain, cshape, t_positions, t_all_d = generate_data(
            data_split, 
            basis_order, 
            n_masses, 
            sample_rate, 
            n_dimensions=n_dimensions, 
            detectors=detectors, 
            window=window, 
            window_acceleration=window_acceleration,
            basis_type=basis_type,
            data_type=data_type,
            fourier_weight=fourier_weight,
            noise_variance=noise_variance)

        #t_label = np.array(labels)[split_ind*data_split : (split_ind + 1)*data_split]
        #t_positions = np.array(positions)[split_ind*data_split : (split_ind + 1)*data_split]
        #t_strain = np.array(strain)[split_ind*data_split : (split_ind + 1)*data_split]

        data_size = len(t_strain)
        t_split_ind = split_ind + start_index

        with h5py.File(os.path.join(data_dir, f"data_{t_split_ind}_{data_size}.hdf5"), "w") as f:
            f.create_dataset("labels", data=np.array(t_labels))
            f.create_dataset("strain", data=np.array(t_strain))
            f.create_dataset("positions", data=np.array(t_positions))

    
def load_data(
    data_dir: str, 
    basis_order: int = 8,
    n_masses: int = 2,
    sample_rate: int = 128,
    n_dimensions: int = 3,
    detectors: list = ["H1", "L1", "V1"],
    window = False,
    window_acceleration = False,
    basis_type = "chebyshev",
    data_type: str = "random"
    ):

    data_path = get_data_path(
        basis_order = basis_order,
        basis_type = basis_type,
        n_masses = n_masses,
        sample_rate = sample_rate,
        n_dimensions = n_dimensions,
        detectors = detectors,
        window = window,
        window_acceleration = False,
        data_type=data_type)

    data_dir = os.path.join(data_dir, data_path)

    with h5py.File(os.path.join(data_dir, "metadata.hdf5"), "r") as f:
        times = np.array(f["times"])
        cshape = np.array(f["poly_order"])
        #basis_type = str(f["basis_type"])

    labels = []
    strain = []
    positions = []

    for fname in os.listdir(data_dir):
        if fname == "metadata.hdf5":
            with h5py.File(os.path.join(data_dir, fname), "r") as f:
                times = np.array(f["times"])
                cshape = np.array(f["poly_order"])
        else:
            with h5py.File(os.path.join(data_dir, fname), "r") as f:
                labels.append(np.array(f["labels"]))
                strain.append(np.array(f["strain"]))
                positions.append(np.array(f["positions"]))


    return times, np.concatenate(labels, axis=0), np.concatenate(strain, axis=0), cshape, np.concatenate(positions, axis=0)

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=False, default="none")
    parser.add_argument("-s", "--datadir", type=str, required=False, default="none")
    parser.add_argument("-ds", "--datasplit", type=int, required=False, default=100000)
    parser.add_argument("-ne", "--nexamples", type=int, required=False, default=100000)
    parser.add_argument("-bo", "--basisorder", type=int, required=False, default=6)
    parser.add_argument("-nm", "--nmasses", type=int, required=False, default=2)
    parser.add_argument("-sr", "--samplerate", type=int, required=False, default=128)
    parser.add_argument("-nd", "--ndimensions", type=int, required=False, default=3)
    parser.add_argument("-ndt", "--ndetectors", type=int, required=False, default=3)
    parser.add_argument("-w", "--window", type=str, required=False, default="none")
    parser.add_argument("-rws", "--returnwindowedcoeffs", type=bool, required=False, default=False)
    parser.add_argument("-bt", "--basis-type", type=str, required=False, default="chebyshev")
    parser.add_argument("-dt", "--data-type", type=str, required=False, default="random")
    parser.add_argument("-fw", "--fourier-weight", type=float, required=False, default=0.0)
    parser.add_argument("-T" "--test-model", type=bool, required=False, default=False)

    args = parser.parse_args()

    dets = ["H1", "L1", "V1"]

    if args.test_model:
        generate_data(
            n_data=5, 
            basis_order=16, 
            n_masses=1, 
            sample_rate=16, 
            n_dimensions = 1, 
            detectors=["H1"], 
            window="none", 
            window_acceleration=True, 
            basis_type="fourier",
            data_type = "kepler",
            fourier_weight=0.0,
            noise_variance=noise_variance)
    else:
        save_data(
            data_dir = args.datadir, 
            data_split = args.datasplit,
            n_examples = args.nexamples,
            basis_order = args.basisorder,
            n_masses = args.nmasses,
            sample_rate = args.samplerate,
            n_dimensions = args.ndimensions,
            detectors = dets[:int(args.ndetectors)],
            window = args.window,
            window_acceleration = args.returnwindowedcoeffs,
            basis_type = args.basis_type,
            data_type = args.data_type,
            fourier_weight = args.fourier_weight
            )

        """
        data = load_data(
            data_dir = args.savedir, 
            basis_order = args.polyorder,
            n_masses = args.nmasses,
            sample_rate = args.samplerate,
            n_dimensions = args.ndimensions,
            detectors = dets[:int(args.ndetectors)],
            window = args.window,
            window_acceleration = args.returnwindowedcoeffs
            )
        """

