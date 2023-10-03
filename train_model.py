import zuko
from data_generation import generate_data, generate_strain_coefficients, generate_2d_derivative
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def train_epoch(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, optimiser: torch.optim, device:str = "cpu", train:bool = True) -> float:
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
        loss = -model(data.flatten(start_dim=1)).log_prob(label).mean()

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
    n_context = config["sample_rate"]
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

    model = zuko.flows.spline.NSF(n_features, context=n_context, transforms=config["ntransforms"], bins=config["nsplines"], hidden_features=config["hidden_features"]).to(config["device"])
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_losses = []
    val_losses = []

    print("Start training")
    for epoch in range(config["n_epochs"]):

        train_loss = train_epoch(train_loader, model, optimiser, device=config["device"], train=True)
        train_losses.append(train_loss)

        with torch.no_grad():
            val_loss = train_epoch(test_loader, model, optimiser, device=config["device"], train=False)
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
        test_model_2d(model, test_loader, times, config["n_masses"], config["chebyshev_order"], config["n_dimensions"], config["root_dir"], config["device"])
    
    print("Completed Testing")

def get_dynamics(coeffmass_samples, times, n_masses, chebyshev_order, n_dimensions):

    masses = coeffmass_samples[-n_masses:]
    coeffs = coeffmass_samples[:-n_masses].reshape(n_masses,chebyshev_order, n_dimensions)

    tseries = np.zeros((n_masses, n_dimensions, len(times)))
    for mass_index in range(n_masses):
        for dim_index in range(n_dimensions):
            tseries[mass_index, dim_index] = np.polynomial.chebyshev.chebval(times, coeffs[mass_index, dim_index, :])

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

def make_2d_animation(root_dir, index, timeseries, masses):

    n_frames = np.shape(timeseries)[-1]
    num_masses = len(masses)

    # Create a figure and axis
    fig, ax = plt.subplots()

    ax.set_xlim([min(timeseries[:,0,:]),max(timeseries[:,0,:])])
    ax.set_ylim([min(timeseries[:,1,:]),max(timeseries[:,1,:])])

    # Create particles as lines
    particles = [ax.plot(0, 0, marker="o", markersize=masses[mind]*8) for mind in range(num_masses)]

    def update_plot(frame):
        for mind in range(num_masses):
            # Set new positions for each particle based on the current frame
            x, y = timeseries[mind][:, frame]
            particles[mind][0].set_data(x, y)


    ani = animation.FuncAnimation(fig, update_plot, frames=n_frames, interval=1)

    writergif = animation.PillowWriter(fps=30) 
    ani.save(os.path.join(root_dir, f"animation_{index}.gif"), writer=writergif)


def test_model_2d(model, dataloader, times, n_masses, chebyshev_order, n_dimensions, root_dir, device):

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
            ax[3].plot(times, data[0].cpu().numpy(), label="source data")

            fig.savefig(os.path.join(plot_out, f"reconstructed_{batch}.png"))

            make_2d_animation(plot_out, batch, recon_tseries, recon_masses)

if __name__ == "__main__":

    config = dict(
        n_data = 100000,
        batch_size = 512,
        chebyshev_order = 6,
        n_masses = 2,
        n_dimensions = 2,
        sample_rate = 128,
        n_epochs = 1000,
        device = "cuda:0",
        nsplines = 6,
        ntransforms = 6,
        hidden_features = [256,256,256],
        root_dir = "test_model_7"
    )

    train_model(config)