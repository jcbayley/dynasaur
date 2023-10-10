import zuko
from data_generation import generate_data, generate_strain_coefficients, generate_2d_derivative
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def train_epoch(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, pre_model: torch.nn.Module, optimiser: torch.optim, device:str = "cpu", train:bool = True) -> float:
    """train one epoch for data

    Args:
        dataloader (torch.DataLoader): dataloader object for all data
        model (torch.Module): pytorch model
        optimiser (torch.optim): pytorch optimiser
        device (str, optional): which device to place model and data Defaults to "cpu".
        train (bool, optional): whether to update model weights or not. Defaults to True.

    Returns:
        float: the average loss over the epoch
    """
    if train:
        model.train()
    else:
        model.eval()

    train_loss = 0
    for batch, (label, data) in enumerate(dataloader):
        label, data = label.to(device), data.to(device)

        optimiser.zero_grad()
        input_data = pre_model(data)
        loss = -model(input_data).log_prob(label).mean()

        if train:
            loss.backward()
            optimiser.step()

        train_loss += loss.item()

    return train_loss/len(dataloader)

def train_model(config: dict) -> None:
    """_summary_

    Args:
        root_dir (str): _description_
        config (dict): _description_
    """
    if not os.path.isdir(config["root_dir"]):
        os.makedirs(config["root_dir"])

    n_features = config["chebyshev_order"]*config["n_masses"]*config["n_dimensions"] + config["n_masses"]
    n_context = config["sample_rate"]*2
    print("init", n_features, n_context)
    times, labels, strain = generate_data(config["n_data"], config["chebyshev_order"], config["n_masses"], config["sample_rate"], n_dimensions=config["n_dimensions"])

    fig, ax = plt.subplots()
    ax.plot(strain[0])
    fig.savefig(os.path.join(config["root_dir"], "test_data.png"))


    print(np.shape(labels), np.shape(strain))
    dataset = TensorDataset(torch.Tensor(labels), torch.Tensor(strain))
    train_size = int(0.9*config["n_data"])
    test_size = 10
    train_set, val_set, test_set = random_split(dataset, (train_size, config["n_data"] - train_size - test_size, test_size))
    train_loader = DataLoader(train_set, batch_size=config["batch_size"],shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"])
    test_loader = DataLoader(test_set, batch_size=1)

    pre_model = nn.Sequential(
        nn.Conv1d(1, 16, 8, padding="same"),
        nn.ReLU(),
        nn.Conv1d(16, 16, 4),
        nn.ReLU(),
        nn.Conv1d(16, 16, 4),
        nn.ReLU(),
        nn.Flatten(),
        nn.LazyLinear(n_context)
    ).to(config["device"])

    model = zuko.flows.spline.NSF(n_features, context=n_context, transforms=config["ntransforms"], bins=config["nsplines"], hidden_features=config["hidden_features"]).to(config["device"])
    
    optimiser = torch.optim.AdamW(list(model.parameters()) + list(pre_model.parameters()), lr=config["learning_rate"])

    train_losses = []
    val_losses = []

    print("Start training")
    for epoch in range(config["n_epochs"]):

        train_loss = train_epoch(train_loader, model, pre_model, optimiser, device=config["device"], train=True)
        train_losses.append(train_loss)

        with torch.no_grad():
            val_loss = train_epoch(test_loader, model, pre_model, optimiser, device=config["device"], train=False)
            val_losses.append(val_loss)
            
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Train loss: {train_loss}, Val loss: {val_loss}")

            torch.save({
                "epoch":epoch,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict":optimiser.state_dict(),
            },
            os.path.join(config["root_dir"],"test_model.pt"))

        fig, ax = plt.subplots()
        ax.plot(train_losses)
        ax.plot(val_losses)
        fig.savefig(os.path.join(config["root_dir"], "lossplot.png"))

    print("Completed Training")


    if config["n_dimensions"] == 1:
        test_model_1d(model, test_loader, times, config["n_masses"], config["chebyshev_order"], config["n_dimensions"], config["root_dir"], config["device"])
    elif config["n_dimensions"] == 2:
        test_model_2d(model, pre_model, test_loader, times, config["n_masses"], config["chebyshev_order"], config["n_dimensions"], config["root_dir"], config["device"])
    
    print("Completed Testing")

def get_dynamics(coeffmass_samples, times, n_masses, chebyshev_order, n_dimensions):
    """_summary_

    Args:
        coeffmass_samples (_type_): _description_
        times (_type_): _description_
        n_masses (_type_): _description_
        chebyshev_order (_type_): _description_
        n_dimensions (_type_): _description_

    Returns:
        _type_: _description_
    """
    #print("msshape", np.shape(coeffmass_samples))
    masses = coeffmass_samples[-n_masses:]
    coeffs = coeffmass_samples[:-n_masses].reshape(n_masses,chebyshev_order, n_dimensions)

    tseries = np.zeros((n_masses, n_dimensions, len(times)))
    for mass_index in range(n_masses):
        for dim_index in range(n_dimensions):
            tseries[mass_index, dim_index] = np.polynomial.chebyshev.chebval(times, coeffs[mass_index, :, dim_index])

    return coeffs, masses, tseries

def test_model_1d(model, dataloader, times, n_masses, chebyshev_order, n_dimensions, root_dir, device):

    plot_out = os.path.join(root_dir, "testout")
    if not os.path.isdir(plot_out):
        os.makedirs(plot_out)

    model.eval()
    with torch.no_grad():
        for batch, (label, data) in enumerate(dataloader):
            label, data = label.to(device), data.to(device)
            coeffmass_samples = model(data.flatten(start_dim=1)).sample().cpu().numpy()

            source_coeffs, source_masses, source_tseries = get_dynamics(label[0].cpu().numpy(), times, n_masses, chebyshev_order, n_dimensions)
            recon_coeffs, recon_masses, recon_tseries = get_dynamics(coeffmass_samples[0], times, n_masses, chebyshev_order, n_dimensions)

            fig, ax = plt.subplots(nrows = 4)
            for mass_index in range(n_masses):
                ax[0].plot(times, source_tseries[mass_index, 0])
                ax[1].plot(times, recon_tseries[mass_index, 0])
                ax[2].plot(times, source_tseries[mass_index, 0] - recon_tseries[mass_index, 0])
    
            recon_weighted_coeffs = np.sum(recon_coeffs[:,0] * recon_masses[:, None], axis=0)
            source_weighted_coeffs = np.sum(source_coeffs[:,0] * source_masses[:, None], axis=0)

            recon_strain_coeffs = generate_strain_coefficients(recon_weighted_coeffs)
            source_strain_coeffs = generate_strain_coefficients(source_weighted_coeffs)

            recon_strain = np.polynomial.chebyshev.chebval(times, recon_strain_coeffs)
            source_strain = np.polynomial.chebyshev.chebval(times, source_strain_coeffs)


            ax[3].plot(times, recon_strain, label="recon")
            ax[3].plot(times, source_strain, label="source")
            ax[3].plot(times, data[0][0].cpu().numpy(), label="source data")

            fig.savefig(os.path.join(plot_out, f"reconstructed_{batch}.png"))

def make_2d_animation(root_dir, index, timeseries, masses, true_timeseries, true_masses):
    """_summary_

    Args:
        root_dir (_type_): _description_
        index (_type_): _description_
        timeseries (_type_): _description_
        masses (_type_): _description_
        true_timeseries (_type_): _description_
        true_masses (_type_): _description_
    """
    n_frames = np.shape(timeseries)[-1]
    num_masses = len(masses)

    # Create a figure and axis
    fig, ax = plt.subplots()

    ax.set_xlim([np.min(timeseries[:,0,:]),np.max(timeseries[:,0,:])])
    ax.set_ylim([np.min(timeseries[:,1,:]),np.max(timeseries[:,1,:])])


    # Create particles as lines
    particles = [ax.plot(0, 0, marker="o", markersize=masses[mind]*10) for mind in range(num_masses)]

    true_particles = [ax.plot(0, 0, marker="o", markersize=true_masses[mind]*10, color="k") for mind in range(num_masses)]

    def update_plot(frame):
        for mind in range(num_masses):
            # Set new positions for each particle based on the current frame
            x, y = timeseries[mind][:, frame]
            particles[mind][0].set_data(x, y)

            xt, yt = true_timeseries[mind][:, frame]
            true_particles[mind][0].set_data(xt, yt)


    ani = animation.FuncAnimation(fig, update_plot, frames=n_frames, interval=1)

    writergif = animation.PillowWriter(fps=30) 
    ani.save(os.path.join(root_dir, f"animation_{index}.gif"), writer=writergif)

def make_2d_distribution(root_dir, index, timeseries, masses, true_timeseries, true_masses):
    """_summary_

    Args:
        root_dir (_type_): _description_
        index (_type_): _description_
        timeseries (_type_): _description_
        masses (_type_): _description_
        true_timeseries (_type_): _description_
        true_masses (_type_): _description_
    """
    n_frames = np.shape(timeseries)[-1]
    num_masses = len(masses[0])

    # Create a figure and axis
    fig, ax = plt.subplots()

    ax.set_xlim([np.min(timeseries[:,0,:]),np.max(timeseries[:,0,:])])
    ax.set_ylim([np.min(timeseries[:,1,:]),np.max(timeseries[:,1,:])])


    # Create particles as lines
    #particles = [ax.plot(timeseries[:,mind,0,0], timeseries[:,mind,1,0], marker="o", ls="none",markersize=masses[0,mind]*10) for mind in range(num_masses)]
    particles = [ax.scatter(timeseries[:,mind,0,0], timeseries[:,mind,1,0],s=masses[:,mind]*10) for mind in range(num_masses)]

    true_particles = [ax.plot(true_timeseries[0,0,:], true_timeseries[0,1,:], marker="o", markersize=true_masses[mind]*10, color="k") for mind in range(num_masses)]

    def update_plot(frame):
        for mind in range(num_masses):
            # Set new positions for each particle based on the current frame
            #print(np.shape(timeseries[:,mind][:,:,frame]))
            x, y = timeseries[:,mind][:,:,frame].reshape(2, len(timeseries))
            #particles[mind][0].set_data(x, y)
            particles[mind].set_offsets(np.c_[x, y])

            xt, yt = true_timeseries[mind][:,frame]
            true_particles[mind][0].set_data(xt, yt)


    ani = animation.FuncAnimation(fig, update_plot, frames=n_frames, interval=1)

    writergif = animation.PillowWriter(fps=30) 
    ani.save(os.path.join(root_dir, f"multi_animation_{index}.gif"), writer=writergif)

def test_model_2d(model, pre_model, dataloader, times, n_masses, chebyshev_order, n_dimensions, root_dir, device):
    """_summary_

    Args:
        model (_type_): _description_
        pre_model (_type_): _description_
        dataloader (_type_): _description_
        times (_type_): _description_
        n_masses (_type_): _description_
        chebyshev_order (_type_): _description_
        n_dimensions (_type_): _description_
        root_dir (_type_): _description_
        device (_type_): _description_
    """
    plot_out = os.path.join(root_dir, "testout")
    if not os.path.isdir(plot_out):
        os.makedirs(plot_out)

    model.eval()
    with torch.no_grad():
        for batch, (label, data) in enumerate(dataloader):
            label, data = label.to(device), data.to(device)
            input_data = pre_model(data)
            coeffmass_samples = model(input_data).sample().cpu().numpy()

            print(np.shape(coeffmass_samples[0]))
            source_coeffs, source_masses, source_tseries = get_dynamics(label[0].cpu().numpy(), times, n_masses, chebyshev_order, n_dimensions)
            recon_coeffs, recon_masses, recon_tseries = get_dynamics(coeffmass_samples[0], times, n_masses, chebyshev_order, n_dimensions)

            fig, ax = plt.subplots(nrows = 4)
            for mass_index in range(n_masses):
                ax[0].plot(times, source_tseries[mass_index,0], color="k", alpha=0.8)
                ax[0].plot(times, source_tseries[mass_index,1], color="r", alpha=0.8)
                ax[1].plot(times, recon_tseries[mass_index, 0])
                ax[1].plot(times, recon_tseries[mass_index, 1])
                ax[2].plot(times, source_tseries[mass_index,0] - recon_tseries[mass_index,0])
    
            recon_weighted_coeffs = np.sum(recon_coeffs * recon_masses[:, None, None], axis=0)
            source_weighted_coeffs = np.sum(source_coeffs * source_masses[:, None, None], axis=0)

            recon_strain_tensor = generate_2d_derivative(recon_weighted_coeffs, times)
            source_strain_tensor = generate_2d_derivative(source_weighted_coeffs, times)

            recon_strain = recon_strain_tensor[0,0] + recon_strain_tensor[0,1]
            source_strain = source_strain_tensor[0,0] + source_strain_tensor[0,1]

            ax[3].plot(times, recon_strain, label="recon")
            ax[3].plot(times, source_strain, label="source")
            ax[3].plot(times, data[0][0].cpu().numpy(), label="source data")

            fig.savefig(os.path.join(plot_out, f"reconstructed_{batch}.png"))

            make_2d_animation(plot_out, batch, recon_tseries, recon_masses, source_tseries, source_masses)


            nsamples = 50
            multi_coeffmass_samples = model(input_data).sample((nsamples, )).cpu().numpy()

            #print("multishape", multi_coeffmass_samples.shape)
            m_recon_tseries, m_recon_masses = np.zeros((nsamples, n_masses, n_dimensions, len(times))), np.zeros((nsamples, n_masses))
            for i in range(nsamples):
                #print(np.shape(multi_coeffmass_samples[i]))
                t_co, t_mass, t_time = get_dynamics(multi_coeffmass_samples[i][0], times, n_masses, chebyshev_order, n_dimensions)
                m_recon_tseries[i] = t_time
                m_recon_masses[i] = t_mass

            make_2d_distribution(plot_out, batch, m_recon_tseries, m_recon_masses, source_tseries, source_masses)

def test_model_chirp(root_dir):
    """Simulate a chirp and reconstruct the masses

    Args:
        root_dir (_type_): _description_
    """
    A = 0.2
    f = 10
    fdot = 2

    data = A * t * np.sin(2*np.pi*(f*t + 0.5*fdot*t*t))

    fig, ax = plt.subplots()
    ax.plot(t, data)
    fig.savefig(os.path.join(root_dir,"./chirpwave.png"))


if __name__ == "__main__":

    config = dict(
        n_data = 500,
        batch_size = 512,
        chebyshev_order = 6,
        n_masses = 2,
        n_dimensions = 2,
        sample_rate = 128,
        n_epochs = 100,
        learning_rate = 2e-4,
        device = "cuda:0",
        nsplines = 6,
        ntransforms = 6,
        hidden_features = [256,256,256],
        root_dir = "test_model_12"
    )

    train_model(config)