import zuko
from dynasaur.data_generation import (
    data_generation,
    compute_waveform,
    data_processing
)
from dynasaur.basis_functions import basis
import dynasaur.create_model as create_model
from dynasaur.config import read_config
from scipy import signal
from dynasaur.test_model_timestep import run_testing
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from dynasaur.plotting import plotting
import json
import argparse
import copy

def train_epoch(
    dataloader: torch.utils.data.DataLoader, 
    model: torch.nn.Module, 
    pre_model: torch.nn.Module, 
    optimiser: torch.optim, 
    device:str = "cpu", 
    train:bool = True,
    flow_package="zuko",
    include_previous_positions=False) -> float:
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
    for batch, (label, data, times, previous_positions) in enumerate(dataloader):
        label, data, times, previous_positions = label.to(device), data.to(device), times.to(device), previous_positions.to(device)

        optimiser.zero_grad()
        input_data = pre_model(data)
        # concatenate the time of the sample to the embedded data
        #print(input_data.size(), times.size())
        if include_previous_positions:
            input_data = torch.cat([input_data, times.unsqueeze(-1), previous_positions.flatten(start_dim=1)], dim=-1).to(torch.float32)
        else:
            input_data = torch.cat([input_data, times.unsqueeze(-1)], dim=-1)
        
        if flow_package == "zuko":
            loss = -model(input_data).log_prob(label).mean()
        elif flow_package == "glasflow":
            loss = -model.log_prob(label, conditional=input_data).mean()
        else:
            raise Exception(f"Package {flow_package} not supported")

        if train:
            loss.backward()
            optimiser.step()

        train_loss += loss.item()

    return train_loss/len(dataloader)

    

def run_training(config: dict, continue_train:bool = False) -> None:
    """ run the training loop 

    Args:
        root_dir (str): _description_
        config (dict): _description_
    """
    if not os.path.isdir(config.get("General","root_dir")):
        os.makedirs(config.get("General","root_dir"))

    config.write_to_file(os.path.join(config.get("General", "root_dir"), "config.ini"))

    device = torch.device(config.get("Training", "device"))
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(device))

    data_dimensions = 3


    ########################
    ## Loading and generating data
    #########################

    if config.get("Data","load_data"):
        print("loading data ........")
        times, basis_dynamics, masses, strain, cshape, positions, snrs = data_generation.load_data(
            data_dir = config.get("General","data_dir"), 
            basis_order = config.get("Data","basis_order"),
            n_masses = config.get("Data","n_masses"),
            sample_rate = config.get("Data","sample_rate"),
            n_dimensions = data_dimensions,
            detectors = config.get("Data","detectors"),
            window_strain = config.get("Data","window_strain"),
            window_acceleration = config.get("Data","window_acceleration"),
            basis_type = config.get("Data","basis_type"),
            data_type = config.get("Data","data_type"),
            noise_variance=config.get("Data","noise_variance"),
            return_velocities=config.get("Data","return_velocities"),
            return_accelerations=config.get("Data","return_accelerations")
            )
   
        #config.get("Data","n_data") = len(labels)
    else:
        print("making data ........")
        times, basis_dynamics, masses, strain, cshape, positions, all_dynamics, snrs, basis_velocities, basis_accelerations = data_generation.generate_data(
            n_data=config.get("Training","n_train_data")+ config.get("Training", "n_val_data"), 
            basis_order=config.get("Data","basis_order"), 
            n_masses=config.get("Data","n_masses"), 
            sample_rate=config.get("Data","sample_rate"), 
            n_dimensions=data_dimensions, 
            detectors=config.get("Data","detectors"), 
            window_strain=config.get("Data","window_strain"), 
            window_acceleration=config.get("Data","window_acceleration"),
            basis_type = config.get("Data","basis_type"),
            data_type = config.get("Data","data_type"),
            fourier_weight=config.get("Data","fourier_weight"),
            noise_variance=config.get("Data","noise_variance"),
            snr=config.get("Data","snr"),
            prior_args=config.get("Data","prior_args"),
            return_velocities=config.get("Data","return_velocities"),
            return_accelerations=config.get("Data","return_accelerations"))

    acc_basis_order = cshape

    #window = signal.windows.tukey(np.shape(strain)[-1], alpha=0.5)
    #strain = strain * window[None, None, :]
    print("data generated, preprocessing .... ")
    n_features = cshape*config.get("Data","n_masses")*config.get("Data","n_dimensions") + config.get("Data","n_masses")
    n_context = config.get("FlowNetwork","n_context")
    flow_package = config.get("FlowNetwork","flow_model_type").split("-")[0]

    fig, ax = plt.subplots()
    ax.plot(strain[0])
    fig.savefig(os.path.join(config.get("General","root_dir"), "test_data.png"))

    ########################
    ## Preprocessing data
    #########################

    if continue_train:
        pre_model, model, weights = create_model.load_models(config, device=config.get("Training","device"))
        pre_model, labels, strain, batch_times, previous_positions = data_processing.preprocess_data(
            pre_model, 
            basis_dynamics,
            masses, 
            strain, 
            window_strain=config.get("Data","window_strain"), 
            spherical_coords=config.get("Data","spherical_coords"), 
            initial_run=False,
            n_masses=config.get("Data","n_masses"),
            device=config.get("Training","device"),
            basis_type=config.get("Data","basis_type"),
            n_dimensions=config.get("Data","n_dimensions"),
            split_data=True,
            basis_velocities=basis_velocities,
            basis_accelerations=basis_accelerations,
            n_previous_positions=config.get("Data","n_previous_positions"))
    else:   
        pre_model, model = create_model.create_models(config, device=config.get("Training","device"))
        pre_model.to(config.get("Training","device"))
        model.to(config.get("Training","device"))
        pre_model, labels, strain, batch_times, previous_positions = data_processing.preprocess_data(
            pre_model, 
            basis_dynamics,
            masses, 
            strain, 
            window_strain=config.get("Data","window_strain"), 
            spherical_coords=config.get("Data","spherical_coords"), 
            initial_run=True,
            n_masses=config.get("Data","n_masses"),
            device=config.get("Training","device"),
            basis_type=config.get("Data","basis_type"),
            n_dimensions=config.get("Data","n_dimensions"),
            split_data=True,
            basis_velocities=basis_velocities,
            basis_accelerations=basis_accelerations,
            n_previous_positions=config.get("Data","n_previous_positions"))

    ########################
    ## Splitting data and creatings torch datasets
    #########################

    print("Preprocessing complete, splitting data ........")
    indices = np.random.choice(np.arange(len(positions)), size=10)
    plotting.plot_data(times, positions[indices], strain[indices], 10, config.get("General","root_dir"))

    n_time_samples = config.get("Data", "duration")*config.get("Data", "sample_rate")
    if config.get("Data", "timestep-predict"):
        nkeepsamps = config.get("Training", "timestep-ntrainsamps")
        # get random time samples
        #rints = torch.tensor([torch.randperm(n_time_samples)[:nkeepsamps] + i for i in range(config.get("Training", "n_train_data") + config.get("Training", "n_val_data"))])
        rints = torch.arange(len(labels))
        lbs_item = torch.from_numpy(labels).to(torch.float32)[rints]
        str_item = torch.Tensor(strain)[rints]
        bt_items = torch.Tensor(batch_times)[rints]
        pp_items = torch.Tensor(previous_positions)[rints]
        split_index = config.get("Training", "n_train_data")*nkeepsamps - nkeepsamps*config.get("Training", "n_val_data")
        train_set = TensorDataset(lbs_item[:split_index], str_item[:split_index], bt_items[:split_index], pp_items[:split_index])
        val_set = TensorDataset(lbs_item[split_index:], str_item[split_index:], bt_items[split_index:], pp_items[split_index:])
        #train_set, val_set = random_split(dataset, (config.get("Training", "n_train_data")*nkeepsamps, config.get("Training", "n_val_data")*nkeepsamps))
    else:
        dataset = TensorDataset(torch.from_numpy(labels).to(torch.float32), torch.Tensor(strain), torch.Tensor(batch_times), torch.Tensor(previous_positions))
        train_set, val_set = random_split(dataset, (config.get("Training", "n_train_data"), config.get("Training", "n_val_data")))
    
    train_loader = DataLoader(train_set, batch_size=config.get("Training","batch_size"),shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.get("Training", "batch_size"))

    ########################
    ## Training
    #########################

    optimiser = torch.optim.AdamW(list(model.parameters()) + list(pre_model.parameters()), lr=config.get("Training","learning_rate"))


    if continue_train:
        with open(os.path.join(config.get("General","root_dir"), "train_losses.txt"), "r") as f:
            losses = np.loadtxt(f)
        train_losses = list(losses[0])
        val_losses = list(losses[1])
        start_epoch = len(train_losses)

        optimiser.load_state_dict(weights["optimiser_state_dict"])
    else:
        train_losses = []
        val_losses = []
        start_epoch = 0

    print("Start training")
    for epoch in range(config.get("Training","n_epochs")):
        if continue_train:
            epoch = epoch + start_epoch

        train_loss = train_epoch(train_loader, model, pre_model, optimiser, device=config.get("Training","device"), train=True, flow_package=flow_package, include_previous_positions=config.get("Data","include_previous_positions"))
        train_losses.append(train_loss)

        with torch.no_grad():
            val_loss = train_epoch(val_loader, model, pre_model, optimiser, device=config.get("Training","device"), train=False, flow_package=flow_package, include_previous_positions=config.get("Data","include_previous_positions"))
            val_losses.append(val_loss)
            
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Train loss: {train_loss}, Val loss: {val_loss}")

            with open(os.path.join(config.get("General","root_dir"), "train_losses.txt"), "w") as f:
                np.savetxt(f, [train_losses, val_losses])

            torch.save({
                "epoch":epoch,
                "model_state_dict": model.state_dict(),
                "pre_model_state_dict": pre_model.state_dict(),
                "optimiser_state_dict":optimiser.state_dict(),
                "norm_factor": pre_model.norm_factor,
                "label_norm_factor": pre_model.label_norm_factor,
                "mass_norm_factor": pre_model.mass_norm_factor
            },
            os.path.join(config.get("General","root_dir"),"test_model.pt"))

        fig, ax = plt.subplots(nrows=2)
        ax[0].plot(train_losses)
        ax[0].plot(val_losses)
        ax[1].plot(train_losses)
        ax[1].plot(val_losses)
        ax[1].set_xscale("log")
        ax[1].set_xlabel("Time")
        ax[0].set_ylabel("Loss")
        ax[1].set_ylabel("Loss")
        fig.savefig(os.path.join(config.get("General","root_dir"), "lossplot.png"))

    print("Completed Training")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=False, default="none")
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--continuetrain', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--makeplots', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ntest", type=int, required=False, default=10)
    args = parser.parse_args()

    config = read_config(os.path.abspath(args.config))

    continue_train = args.continuetrain
    train_model = args.train
    test_model = args.test
    print("makeplots", args.makeplots)
        
    if train_model:
        run_training(config, continue_train=continue_train)

    if test_model:
        run_testing(config, make_plots=args.makeplots, n_test=args.ntest)
