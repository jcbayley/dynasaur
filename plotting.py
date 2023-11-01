import matplotlib.pyplot as plt
import numpy as np
import os

def plot_data(times, positions, strain, n_examples, root_dir):
    """plot some examples of data

    Args:
        times (_type_): _description_
        labels (_type_): _description_
        strain (_type_): _description_
        n_examples (_type_): _description_
        root_dir (_type_): _description_
    """
    savedir = os.path.join(root_dir, "train_data")
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

    n_examples = min(n_examples, len(positions))


    n_dim = np.shape(positions)[2]
    n_det = np.shape(strain)[1]

    for i in range(n_examples):
        fig, ax = plt.subplots(nrows=n_dim)
        for dim in range(n_dim):
            ax[dim].plot(times, positions[i, :, dim, :].T)

        fig.savefig(os.path.join(savedir, f"output_positions_{i}.png"))

    
        fig, ax = plt.subplots()
        for det in range(n_det):
            ax.plot(times, strain[i, det, :])

        fig.savefig(os.path.join(savedir, f"output_strain_{i}.png"))

def plot_z_projection(source_tseries, recon_tseries, fname = "z_projection.png"):

    n_masses, n_dimensions, n_samples = np.shape(source_tseries)
    fig, ax = plt.subplots()
    for j in range(n_masses):
        ax.plot(source_tseries[j, 0, :],source_tseries[j, 1, :], color=f"C{j}", label="truth")
        ax.plot(recon_tseries[j, 0, :], recon_tseries[j, 1, :], color=f"C{j}", ls ="--", label="prediction")

    ax.set_ylabel(f"y position")
    ax.set_ylabel(f"x position")

    if fname is not None:
        fig.savefig(fname)

def plot_reconstructions(
    times, 
    detectors, 
    recon_strain, 
    source_strain, 
    source_data, 
    fname:str = None):
    """_summary_

    Args:
        times (_type_): _description_
        detectors (_type_): _description_
        recon_strain (_type_): _description_
        source_strain (_type_): _description_
        source_data (_type_): _description_
        save_image (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    fig, ax = plt.subplots(nrows = 3)
    for i in range(len(detectors)):
        ax[i].plot(times, recon_strain[i], label="recon")
        ax[i].plot(times, source_strain[i], label="source")
        ax[i].plot(times, source_data[i], label="source data")
        ax[i].legend()
        ax[i].set_ylabel(f"{detectors[i]} Strain")

    if fname is not None:
        fig.savefig(fname)
    
    return fig

def plot_positions(
    times, 
    source_tseries, 
    recon_tseries, 
    n_dimensions, 
    n_masses,
    fname:str = None):
    """_summary_

    Args:
        time (_type_): _description_
        source_tseries (_type_): _description_
        recon_tseries (_type_): _description_
        n_dimensions (_type_): _description_
        fname (str, optional): _description_. Defaults to None.
    """
    fig, ax = plt.subplots(nrows=n_dimensions)
    for i in range(n_dimensions):
        for j in range(n_masses):
            ax[i].plot(times, source_tseries[j, i, :], color=f"C{j}")
            ax[i].plot(times, recon_tseries[j, i, :], ls="--", color=f"C{j}")
            ax[i].set_ylabel(f"Dimension {i} position")

    if fname is not None:
        fig.savefig(fname)

def plot_sample_separations(
    times, 
    source_tseries, 
    recon_tseries,  
    fname="./separations.png"):

    n_masses, n_dimensions, n_samples = np.shape(source_tseries)
    separations = []
    for j in range(n_masses):
        sep = recon_tseries[:, j, :, :] - np.expand_dims(source_tseries[j, :, :], 0)
        sep = np.sqrt(np.sum(sep**2, axis=1))
        separations.append(sep)

    fig, ax = plt.subplots()
    for j in range(n_masses):
        ax.plot(times, separations[j].T)


    if fname is not None:
        fig.savefig(fname)

def plot_mass_distributions(
    recon_masses,
    source_masses,
    fname="massdist.png"):

    n_samples, n_masses = np.shape(recon_masses)

    fig, ax = plt.subplots()

    for j in range(n_masses):
        massdiff = recon_masses[:,j] - source_masses[j]
        ax.hist(massdiff, bins=20, alpha=0.6)

    if fname is not None:
        fig.savefig(fname)

