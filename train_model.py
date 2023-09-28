import zuko
from data_generation import generate_data, generate_strain_coefficients
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def train_batch(dataloader, model, optimiser, device="cpu", train=True):

    if train:
        model.train()
    else:
        model.eval()

    train_loss = 0
    for batch, (label, data) in enumerate(dataloader):
        label, data = label.to(device), data.to(device)

        optimiser.zero_grad()
        loss = -model(data).log_prob(label).mean()

        if train:
            loss.backward()
            optimiser.step()

        train_loss += loss.item()

    return train_loss/len(dataloader)

def train_model(root_dir):

    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    n_data = 100000
    batch_size = 512
    chebyshev_order = 10
    n_masses = 3
    n_dimensions = 1
    sample_rate = 128
    n_epochs = 5000
    device = "cuda:0"

    n_features = chebyshev_order*n_masses*n_dimensions + n_masses
    n_context = sample_rate

    times, labels, strain = generate_data(n_data, chebyshev_order, n_masses, sample_rate, n_dimensions=n_dimensions)

    fig, ax = plt.subplots()
    ax.plot(strain[0])
    fig.savefig(os.path.join(root_dir, "test_data.png"))


    dataset = TensorDataset(torch.Tensor(labels), torch.Tensor(strain))
    train_size = int(0.9*n_data)
    test_size = 10
    train_set, val_set, test_set = random_split(dataset, (train_size, n_data - train_size - test_size, test_size))
    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=1)

    model = zuko.flows.spline.NSF(n_features, context=n_context, bins=5, hidden_features=[128,128,128]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_losses = []
    val_losses = []

    print("Start training")
    for epoch in range(n_epochs):

        train_loss = train_batch(train_loader, model, optimiser, device=device, train=True)
        train_losses.append(train_loss)

        with torch.no_grad():
            val_loss = train_batch(test_loader, model, optimiser, device=device, train=False)
            val_losses.append(val_loss)
            
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Train loss: {train_loss}, Val loss: {val_loss}")

            torch.save({
                "epoch":epoch,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict":optimiser.state_dict(),
            },
            os.path.join(root_dir,"test_model.pt"))

    print("Completed Training")
    fig, ax = plt.subplots()
    ax.plot(train_losses)
    ax.plot(val_losses)
    fig.savefig(os.path.join(root_dir, "lossplot.png"))

    if n_dimensions == 1:
        test_model_1d(model, test_loader, times, n_masses, chebyshev_order, root_dir, device)
    elif n_dimensions == 2:
        test_model_2d(model, test_loader, times, n_masses, chebyshev_order, root_dir, device)
    
    print("Completed Testing")

def get_dynamics(coeffmass_samples, times, n_masses, chebyshev_order):

    masses = coeffmass_samples[-n_masses:]
    coeffs = coeffmass_samples[:-n_masses].reshape(n_masses,chebyshev_order)

    tseries = np.zeros((n_masses, len(times)))
    for mass_index in range(n_masses):
        tseries[mass_index] = np.polynomial.chebyshev.chebval(times, coeffs[mass_index, :])

    return coeffs, masses, tseries

def test_model_1d(model, dataloader, times, n_masses, chebyshev_order, root_dir, device):

    plot_out = os.path.join(root_dir, "testout")
    if not os.path.isdir(plot_out):
        os.makedirs(plot_out)

    model.eval()
    with torch.no_grad():
        for batch, (label, data) in enumerate(dataloader):
            label, data = label.to(device), data.to(device)
            coeffmass_samples = model(data).sample().cpu().numpy()

            source_coeffs, source_masses, source_tseries = get_dynamics(label[0].cpu().numpy(), times, n_masses, chebyshev_order)
            recon_coeffs, recon_masses, recon_tseries = get_dynamics(coeffmass_samples[0], times, n_masses, chebyshev_order)

            fig, ax = plt.subplots(nrows = 4)
            for mass_index in range(n_masses):
                ax[0].plot(times, source_tseries[mass_index])
                ax[1].plot(times, recon_tseries[mass_index])
                ax[2].plot(times, source_tseries[mass_index] - recon_tseries[mass_index])
    
            recon_weighted_coeffs = np.sum(recon_coeffs * recon_masses[:, None], axis=0)
            source_weighted_coeffs = np.sum(source_coeffs * source_masses[:, None], axis=0)

            recon_strain_coeffs = generate_strain_coefficients(recon_weighted_coeffs)
            source_strain_coeffs = generate_strain_coefficients(source_weighted_coeffs)

            recon_strain = np.polynomial.chebyshev.chebval(times, recon_strain_coeffs)
            source_strain = np.polynomial.chebyshev.chebval(times, source_strain_coeffs)


            ax[3].plot(times, recon_strain, label="recon")
            ax[3].plot(times, source_strain, label="source")
            ax[3].plot(times, data[0].cpu().numpy(), label="source data")

            fig.savefig(os.path.join(plot_out, f"reconstructed_{batch}.png"))

def test_model_2d(model, dataloader, times, n_masses, chebyshev_order, root_dir, device):

    plot_out = os.path.join(root_dir, "testout")
    if not os.path.isdir(plot_out):
        os.makedirs(plot_out)

    model.eval()
    with torch.no_grad():
        for batch, (label, data) in enumerate(dataloader):
            label, data = label.to(device), data.to(device)
            coeffmass_samples = model(data).sample().cpu().numpy()

            source_coeffs, source_masses, source_tseries = get_dynamics(label[0].cpu().numpy(), times, n_masses, chebyshev_order)
            recon_coeffs, recon_masses, recon_tseries = get_dynamics(coeffmass_samples[0], times, n_masses, chebyshev_order)

            fig, ax = plt.subplots(nrows = 4)
            for mass_index in range(n_masses):
                ax[0].plot(times, source_tseries[mass_index])
                ax[1].plot(times, recon_tseries[mass_index])
                ax[2].plot(times, source_tseries[mass_index] - recon_tseries[mass_index])
    
            recon_weighted_coeffs = np.sum(recon_coeffs * recon_masses[:, None], axis=0)
            source_weighted_coeffs = np.sum(source_coeffs * source_masses[:, None], axis=0)

            recon_strain_coeffs = generate_strain_coefficients(recon_weighted_coeffs)
            source_strain_coeffs = generate_strain_coefficients(source_weighted_coeffs)

            recon_strain = np.polynomial.chebyshev.chebval(times, recon_strain_coeffs)
            source_strain = np.polynomial.chebyshev.chebval(times, source_strain_coeffs)


            ax[3].plot(times, recon_strain, label="recon")
            ax[3].plot(times, source_strain, label="source")
            ax[3].plot(times, data[0].cpu().numpy(), label="source data")

            fig.savefig(os.path.join(plot_out, f"reconstructed_{batch}.png"))

if __name__ == "__main__":

    train_model("./test_model_4")