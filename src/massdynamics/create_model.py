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
import glasflow
from zuko.distributions import DiagNormal
import numpy as np
from massdynamics.data_generation import (
    data_generation,
)

import torch
import torch.nn as nn
import os

class PreNetworkAttention(torch.nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads=2, num_layers=2):
        super(PreNetworkAttention, self).__init__()
        self.embedding = torch.nn.Linear(input_dim, embed_dim)
        encoder_layers = torch.nn.TransformerEncoderLayer(embed_dim, num_heads)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = torch.nn.Linear(embed_dim, output_dim)
        
    def forward(self, x, layer_number=None):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Change to shape (seq_len, batch_size, embed_dim) for Transformer encoder
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Change back to shape (batch_size, seq_len, embed_dim)
        x = torch.mean(x, dim=1)  # Global average pooling
        x = self.fc(x)
        return x

def create_models(config, device=None):
    """create a convolutional to linear model with n_context outputs and a 
    flow model taking in n_context parameters and layers defined in config file

    Args:
        config: config file containing model parameters
        device: which device to put the model on
    Returns:
        tuple of models: (pre_model, model)
    """

    """
    times, basis_dynamics, masses, strain, feature_shape, positions, all_dynamics = data_generation.generate_data(
        2, 
        config["basis_order"], 
        config["n_masses"], 
        config["sample_rate"], 
        n_dimensions=config["n_dimensions"], 
        detectors=config["detectors"], 
        window=config["window"], 
        window_acceleration=config["window_acceleration"],
        basis_type=config["basis_type"],
        data_type=config["data_type"])
    """

    n_basis = config["basis_order"]
    if config["basis_type"] == "fourier":
        n_basis += 2
    
    if config["timestep-predict"]:
        feature_shape = config["n_masses"] + config["n_masses"]*config["n_dimensions"]
    else:
        feature_shape = config["n_masses"] + config["n_masses"]*config["n_dimensions"]*n_basis

    n_features = feature_shape#cshape*config["n_masses"]*config["n_dimensions"] + config["n_masses"]
    n_context = config["n_context"]
    if config["timestep-predict"]:
        n_context += 1
        
    n_input = config["sample_rate"]*config["duration"]

    if device is not None:
        config["device"] = device

    # pre processing creation
    if "transformer_layers" in config:
        if config["transformer_layers"]:

            pre_model = PreNetworkAttention(
                n_input, 
                n_context, 
                config["transformer_layers"]["embed_dim"], 
                num_heads=config["transformer_layers"]["num_heads"], 
                num_layers=config["transformer_layers"]["num_layers"])
        else:
            raise Exception("Please define either transformer or convolution not both")
    else:
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
    if config["flow_model_type"] == "zuko-custom":
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
        
    elif config["flow_model_type"] == "zuko-cnf":
        model = zuko.flows.CNF(
            n_features, 
            context=n_context, 
            hidden_features=config["hidden_features"]
            ).to(config["device"])

    elif config["flow_model_type"] == "zuko_nsf":
        model = zuko.flows.spline.NSF(
            n_features, 
            context=n_context, 
            transforms=config["ntransforms"], 
            bins=config["nsplines"], 
            hidden_features=config["hidden_features"]
            ).to(config["device"])
        
    elif config["flow_model_type"] == "glasflow-nsf":
        model = glasflow.CouplingNSF(
            n_inputs=n_features,
            n_transforms=config["ntransforms"],
            n_blocks_per_transform=len(config["hidden_features"]),
            n_conditional_inputs=n_context,
            n_neurons=config["hidden_features"][0],
            num_bins=config["nsplines"]
        ).to(config["device"])

    elif config["flow_model_type"] == "glasflow-enflow":
        # Not working yet
        model = glasflow.EnFlow(
            n_inputs=n_features,
            n_transforms=config["n_transforms"],
            n_conditional_inputs=n_context,
            n_neurons=config["hidden_features"],
            num_bins=config["nsplines"]
        ).to(config["device"])
    else:
        print("-- No flow specified -- Using zuko nsf --")
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
    times, basis_dynamics, masses, strain, feature_shape, positions, all_dynamics, snr = data_generation.generate_data(
        2, 
        config["basis_order"], 
        config["n_masses"], 
        config["sample_rate"], 
        n_dimensions=3, 
        detectors=config["detectors"], 
        window=config["window"], 
        window_acceleration=config["window_acceleration"],
        basis_type=config["basis_type"],
        data_type=config["data_type"],
        prior_args=config["prior_args"])

    n_basis = config["basis_order"]
    if config["basis_type"] == "fourier":
        n_basis += 2
    feature_shape = config["n_masses"] + config["n_masses"]*config["n_dimensions"]*n_basis

    n_features = feature_shape#cshape*config["n_masses"]*config["n_dimensions"] + config["n_masses"]
    n_context = config["n_context"]
    n_input = config["sample_rate"]*config["duration"]

    pre_model, model = create_models(config, device)

    pre_model.to(device)
    model.to(device)
    
    weights = torch.load(os.path.join(config["root_dir"],"test_model.pt"), map_location=device)

    pre_model.load_state_dict(weights["pre_model_state_dict"])

    model.load_state_dict(weights["model_state_dict"])

    pre_model.norm_factor = weights["norm_factor"]
    pre_model.label_norm_factor = weights["label_norm_factor"]
    pre_model.mass_norm_factor = weights["mass_norm_factor"]
    """
    if "norm_factor" in weights:
        pre_model.norm_factor = weights["norm_factor"]
    else:
        pre_model.norm_factor = 1.0
    """
    return pre_model, model, weights


def backwards_pass(pre_model, model, data):

    input_data = pre_model(data)

    output = model(input_data)

