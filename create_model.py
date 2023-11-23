import zuko
from zuko.transforms import (
    AffineTransform,
    MonotonicRQSTransform,
    RotationTransform,
    SigmoidTransform,
    SoftclipTransform
)
from zuko.flows import (
    Flow,
    GeneralCouplingTransform,
    MaskedAutoregressiveTransform,
    NeuralAutoregressiveTransform,
    Unconditional,
    LazyTransform,
)
from zuko.distributions import DiagNormal
import numpy as np
from data_generation import (
    generate_data, 
    generate_strain_coefficients, 
    compute_strain_from_coeffs, 
    window_coeffs, 
    perform_window, 
    compute_hTT_coeffs,
    basis, 
    compute_energy_loss
)
import torch
import torch.nn as nn
import os


def create_models(config, device):
    """create a convolutional to linear model with n_context outputs and a 
    flow model taking in n_context parameters and layers defined in config file

    Args:
        config: config file containing model parameters
        device: which device to put the model on
    Returns:
        tuple of models: (pre_model, model)
    """

    times, labels, strain, cshape, positions, all_dynamics = generate_data(
        2, 
        config["basis_order"], 
        config["n_masses"], 
        config["sample_rate"], 
        n_dimensions=config["n_dimensions"], 
        detectors=config["detectors"], 
        window=config["window"], 
        return_windowed_coeffs=config["return_windowed_coeffs"],
        basis_type=config["basis_type"],
        data_type=config["data_type"])


    n_features = cshape*config["n_masses"]*config["n_dimensions"] + config["n_masses"]
    n_context = 2*config["sample_rate"]

    # pre processing creation
    pre_model = nn.Sequential()

    for lind, layer in enumerate(config["conv_layers"]):
        pre_model.add_module(f"conv_{lind}", nn.Conv1d(layer[0], layer[1], layer[2], padding="same"))
        pre_model.add_module(f"relu_{lind}", nn.ReLU())
        if layer[3] > 1:
            pre_model.add_module(f"maxpool_{lind}", nn.MaxPool1d(layer[3]))

    pre_model.add_module("flatten", nn.Flatten())
    
    for lind, layer in enumerate(config["linear_layers"]):
        pre_model.add_module(f"lin_{lind}", nn.LazyLinear(layer))

    pre_model.add_module("output", nn.LazyLinear(n_context))

    # Flow creation

    if config["custom_flow"]:
        bins = config["nsplines"]
        randperm = False
        orders = [
            torch.arange(n_features),
            torch.flipud(torch.arange(n_features)),
        ]
        transforms = [
            Unconditional(lambda: SigmoidTransform().inv),
        ]

        for i in range(config["ntransforms"]):
            transforms.append(MaskedAutoregressiveTransform(
                features=n_features,
                context=n_context,
                order=torch.randperm(features) if randperm else orders[i % 2],
                univariate=MonotonicRQSTransform,
                shapes=[(bins,), (bins,), (bins - 1,)],
                hidden_features=config["hidden_features"]
                )
            )
        
        #transforms.append(
        #    Unconditional(lambda: SoftclipTransform(1.0).inv)
        #)

        
        base = Unconditional(
            DiagNormal,
            torch.zeros(n_features),
            torch.ones(n_features),
            buffer=True,
        )
        model = Flow(
            transform=transforms, 
            base=base
            ).to(config["device"])
        
    else:
        model = zuko.flows.spline.NSF(
            n_features, 
            context=n_context, 
            transforms=config["ntransforms"], 
            bins=config["nsplines"], 
            hidden_features=config["hidden_features"]
            ).to(config["device"])

    return pre_model, model

def load_models(config, device):
    """Load in models from config

    Args:
        config (_type_): config dictionary
        device (_type_): which device to put the models on

    Returns:
        tuple: pre_model, model
    """
    times, labels, strain, cshape, positions, all_dynamics = generate_data(
        2, 
        config["basis_order"], 
        config["n_masses"], 
        config["sample_rate"], 
        n_dimensions=config["n_dimensions"], 
        detectors=config["detectors"], 
        window=config["window"], 
        return_windowed_coeffs=config["return_windowed_coeffs"],
        basis_type=config["basis_type"],
        data_type=config["data_type"])

    if config["basis_type"] == "fourier":
        n_features = cshape*config["n_masses"]*config["n_dimensions"] + config["n_masses"]
    else:
        n_features = cshape*config["n_masses"]*config["n_dimensions"] + config["n_masses"]

    n_context = 2*config["sample_rate"]

    pre_model, model = create_models(config, device)

    pre_model.to(device)
    model.to(device)
    
    weights = torch.load(os.path.join(config["root_dir"],"test_model.pt"), map_location=device)

    pre_model.load_state_dict(weights["pre_model_state_dict"])

    model.load_state_dict(weights["model_state_dict"])

    if "norm_factor" in weights:
        pre_model.norm_factor = weights["norm_factor"]
    else:
        pre_model.norm_factor = 1.0

    return pre_model, model

