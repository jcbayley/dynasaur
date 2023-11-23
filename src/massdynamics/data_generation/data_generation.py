import numpy as np
import lal
import lalpulsar 
import matplotlib.pyplot as plt
import scipy.signal as signal
import argparse
import h5py
import os
import torch
from massdynamics.data_generation import newtonian_orbits
from massdynamics.data_generation import kepler_orbits
from massdynamics.data_generation import random_orbits
from massdynamics.basis_functions import basis


def generate_data(
    n_data: int, 
    basis_order: int, 
    n_masses:int, 
    sample_rate: int, 
    n_dimensions: int = 1, 
    detectors=["H1"], 
    window="none", 
    return_windowed_coeffs=True, 
    basis_type="chebyshev",
    data_type = "random",
    fourier_weight=0.0):

    if data_type == "random":
        return random_orbits.generate_data(
                n_data, 
                basis_order, 
                n_masses, 
                sample_rate, 
                n_dimensions, 
                detectors=detectors, 
                window=window, 
                return_windowed_coeffs=return_windowed_coeffs, 
                basis_type=basis_type,
                fourier_weight=fourier_weight)
    elif data_type == "newton":
        return newtonian_orbits.generate_data(
                n_data, 
                basis_order, 
                n_masses, 
                sample_rate, 
                n_dimensions, 
                detectors=detectors, 
                window=window, 
                return_windowed_coeffs=return_windowed_coeffs, 
                basis_type=basis_type)
    elif data_type == "kepler":
        return kepler_orbits.generate_data(
                n_data, 
                basis_order, 
                n_masses, 
                sample_rate, 
                n_dimensions, 
                detectors=detectors, 
                window=window, 
                return_windowed_coeffs=return_windowed_coeffs, 
                basis_type=basis_type)

def get_data_path(
    basis_order: int = 8,
    basis_type: str = "chebyshev",
    n_masses: int = 2,
    sample_rate: int = 128,
    n_dimensions: int = 3,
    detectors: list = ["H1", "L1", "V1"],
    window: str = "none",
    return_windowed_coeffs = False,
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
    return_windowed_coeffs = False,
    basis_type: str = "chebyshev",
    data_type: str = "random",
    start_index: int = 0,
    fourier_weight:float=0.0
    ):


    data_path = get_data_path(
        basis_order = basis_order,
        basis_type = basis_type,
        n_masses = n_masses,
        sample_rate = sample_rate,
        n_dimensions = n_dimensions,
        detectors = detectors,
        window = window,
        return_windowed_coeffs = False,
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
        return_windowed_coeffs=return_windowed_coeffs,
        basis_type=basis_type,
        data_type=data_type,
        fourier_weight=fourier_weight)


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
            return_windowed_coeffs=return_windowed_coeffs,
            basis_type=basis_type,
            data_type=data_type,
            fourier_weight=fourier_weight)

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
    return_windowed_coeffs = False,
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
        return_windowed_coeffs = False,
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

    args = parser.parse_args()

    dets = ["H1", "L1", "V1"]

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
        return_windowed_coeffs = args.returnwindowedcoeffs,
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
        return_windowed_coeffs = args.returnwindowedcoeffs
        )
    """
