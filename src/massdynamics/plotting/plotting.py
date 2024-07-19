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


        fig, ax = plt.subplots()
        ax.plot(positions[i, 0, 0, :], positions[i, 0, 1, :])
        ax.plot(positions[i, 1, 0, :], positions[i, 1, 1, :])

        fig.savefig(os.path.join(savedir, f"xyplane_positions_{i}.png"))

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
    source_energy,
    recon_energy,
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
    fig, ax = plt.subplots(nrows = 4)
    for i in range(len(detectors)):
        ax[i].plot(times, recon_strain[i], label="recon")
        ax[i].plot(times, source_strain[i], label="source", ls="--")
        dattimes = np.linspace(0,np.max(times), len(source_data[i])) 
        ax[i].plot(dattimes, source_data[i], label="source data")
        ax[i].legend()
        ax[i].set_ylabel(f"{detectors[i]} Strain")


    ax[3].plot(times, recon_energy)
    ax[3].plot(times, source_energy)

    if fname is not None:
        fig.savefig(fname)
    
    return fig

def plot_sampled_reconstructions(
    times, 
    detectors, 
    recon_strain, 
    source_strain, 
    fname:str = None):
    """_summary_

    Args:
        times (_type_): timestamps of samples
        detectors (_type_): _list of detectors used
        recon_strain (_type_): reconstructed strain (n_samples, n_detectors, n_timesamples)
        source_strain (_type_): strain (n_detectors, n_timesamples)
        save_image (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """

    lower, med, upper = np.quantile(recon_strain, [0.1, 0.5, 0.9], axis = 0)

    fig, ax = plt.subplots(nrows = 4)
    for i in range(len(detectors)):
        ax[i].fill_between(times, lower[i], upper[i], label="90% confidence", alpha=0.5, color=f"C{i}")
        ax[i].plot(times, med[i], color=f"C{i}", label="median")
        ax[i].plot(times, source_strain[i], label="source", color="k")
        ax[i].legend()
        ax[i].set_ylabel(f"{detectors[i]} Strain")


    #ax[3].plot(times, recon_energy)
    #ax[3].plot(times, source_energy)

    if fname is not None:
        fig.savefig(fname)
    
    return fig

def plot_sample_positions(
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
    lower, med, upper = np.quantile(recon_tseries, [0.1, 0.5, 0.9], axis = 0)
    print(np.shape(med), len(times), np.shape(recon_tseries))
    for i in range(n_dimensions):
        for j in range(n_masses):
            #ax[i].fill_between(times, lower[j,i], upper[j,i], label="90% confidence", alpha=0.5, color=f"C{j}")
            #ax[i].plot(times, med[j,i], color=f"C{j}", label="median")
            ax[i].plot(times, source_tseries[j,i], label="source", color="k")
            #ax[i].plot(times, source_tseries[j, i, :], color=f"C{j}")
            ax[i].plot(times, recon_tseries[:, j, i, :].T, ls="--", color=f"C{j}", alpha=0.4)
            ax[i].set_ylabel(f"Dimension {i} position")

    if fname is not None:
        fig.savefig(fname)


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

    fig, ax = plt.subplots(ncols=n_masses + 1)

    for j in range(n_masses):
        ax[j].hist(recon_masses[:,j], bins=20, alpha=0.6)
        ax[j].axvline(source_masses[j], color="r")
        ax[j].set_xlabel(f"Mass_{j}")

    ax[-1].hist(np.sum(recon_masses, axis=1), bins=20, alpha=0.6)
    ax[-1].axvline(np.sum(source_masses), color="r")
    ax[-1].set_xlabel("Total Mass")

    if fname is not None:
        fig.savefig(fname)

def plot_1d_posteriors(samples, truths, fname=None):
    """_summary_

    Args:
        samples (_type_): shape(nsamples, ndims)
        truths (_type_): (ndims)
    """

    """
    fig, ax = plt.subplots(figsize = (len(truths), 7))
    print(np.shape(samples), np.shape(truths))
    ax.violinplot(samples)
    ax.plot(np.arange(len(truths)) + 1, truths, marker="o", ls="none")
    """
    fig, ax = plt.subplots(nrows = samples.shape[1], figsize = (7, len(truths)))
    for i in range(samples.shape[1]):
        ax[i].hist(samples[:,i], bins=20, alpha=0.7)
        ax[i].axvline(truths[i], color="r")
        ax[i].set_xlabel(f"Index: {i}")
        ax[i].set_ylabel("Count")
    
    fig.tight_layout()
    
    if fname is not None:
        fig.savefig(fname)


def plot_dimension_projection(positions, true_positions, fname=None, alpha=0.5):
    """plot the three projections in dimensions and the third dimension from 45 degrees


    Args:
        positions (_type_): (nsamples, n_masses, n_dimensions, n_times)
        alpha (float, optional): _description_. Defaults to 0.5.
    """
    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax3d = fig.add_subplot(2,2,4, projection="3d")
    fig.delaxes(ax[1,1])
    dind=0
    alpha=0.5
    ax[0,0].set_xlabel("x")
    ax[0,0].set_ylabel("y")
    ax[0,1].set_xlabel("x")
    ax[0,1].set_ylabel("z")
    ax[1,0].set_xlabel("y")
    ax[1,0].set_ylabel("z")
    for massind in range(np.shape(positions)[1]):
        ax[0,0].plot(positions[:, massind, 0].T, positions[:, massind, 1].T, color=f"C{massind}", alpha=alpha)
        ax[0,1].plot(positions[:, massind, 0].T, positions[:, massind, 2].T, color=f"C{massind}", alpha=alpha)
        ax[1,0].plot(positions[:, massind, 1].T, positions[:, massind, 2].T, color=f"C{massind}", alpha=alpha)
        for x,y,z in positions[:, massind, :]:
            ax3d.plot3D(x,y,z, color=f"C{massind}", alpha=alpha)

    for massind in range(np.shape(true_positions)[0]):
        ax[0,0].plot(true_positions[massind, 0].T, true_positions[massind, 1].T, color=f"k")
        ax[0,1].plot(true_positions[massind, 0].T, true_positions[massind, 2].T, color=f"k")
        ax[1,0].plot(true_positions[massind, 1].T, true_positions[massind, 2].T, color=f"k")
        x,y,z = true_positions[massind, :]
        ax3d.plot3D(x,y,z, color=f"k",)

    fig.tight_layout()

    if fname is not None:
        fig.savefig(fname)



