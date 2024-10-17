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
from dynasaur.data_generation import (
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
        config.get("basis_order"], 
        config.get("n_masses"], 
        config.get("sample_rate"], 
        n_dimensions=config.get("n_dimensions"], 
        detectors=config.get("detectors"], 
        window=config.get("window"], 
        window_acceleration=config.get("window_acceleration"],
        basis_type=config.get("basis_type"],
        data_type=config.get("data_type"])
    """

    if device is None:
        device = config.get("Training", "device")

    n_basis = config.get("Data", "basis_order")
    if config.get("Data", "basis_type") == "fourier":
        n_basis += 2

    if config.get("Data", "return_velocities"):
        vel_features = 2
    else:
        vel_features = 1

    if config.get("Data", "n_previous_positions") > 0:
        extra_context = config.get("Data", "n_masses")*config.get("Data", "n_dimensions")*config.get("Data", "n_previous_positions")
    else:
        extra_context = 0

    if config.get("Data", "timestep-predict"):
        tstep_context = 1
    else:
        tstep_context = 0


    if config.get("Data", "timestep-predict"):
        feature_shape = config.get("Data", "n_masses") + config.get("Data", "n_masses")*config.get("Data", "n_dimensions")*vel_features
    else:
        feature_shape = config.get("Data", "n_masses") + config.get("Data", "n_masses")*config.get("Data", "n_dimensions")*n_basis

    n_features = feature_shape#cshape*config["n_masses"]*config["n_dimensions"] + config["n_masses"]
    
    n_context = config.get("FlowNetwork", "n_context") + tstep_context + extra_context

    n_input = config.get("Data", "sample_rate")*config.get("Data", "duration")

    # pre processing creation
    if config.get("PreNetwork", "transformer_layers") not in ["none", None]:
        if config.get("PreNetwork", "transformer_layers"):

            pre_model = PreNetworkAttention(
                n_input, 
                n_context - tstep_context - extra_context, 
                config.get("PreNetwork","transformer_layers")["embed_dim"], 
                num_heads=config.get("PreNetwork", "transformer_layers")["num_heads"], 
                num_layers=config.get("PreNetwork", "transformer_layers")["num_layers"])
        else:
            raise Exception("Please define either transformer or convolution not both")
    elif config.get("PreNetwork", "conv_layers") not in ["none", None]:
        pre_model = nn.Sequential()

        for lind, layer in enumerate(config.get("PreNetwork","conv_layers")):
            pre_model.add_module(f"conv_{lind}", nn.Conv1d(layer[0], layer[1], layer[2], padding="same"))
            pre_model.add_module(f"relu_{lind}", nn.ReLU())
            if layer[3] > 1:
                pre_model.add_module(f"maxpool_{lind}", nn.MaxPool1d(layer[3]))

        pre_model.add_module("flatten", nn.Flatten())
        
        for lind, layer in enumerate(config.get("PreNetwork", "linear_layers")):
            pre_model.add_module(f"lin_{lind}", nn.LazyLinear(layer))

        pre_model.add_module("output", nn.LazyLinear(n_context))
    else:
        raise Exception("No Pre network parameters")

    # Flow creation
    if config.get("FlowNetwork", "flow_model_type") == "zuko-custom":
        bins = config.get("FlowNetwork", "nsplines")
        randperm = False
        orders = [
            torch.arange(n_features),
            torch.flipud(torch.arange(n_features)),
        ]
        transforms = [
            Unconditional(lambda: SigmoidTransform().inv),
        ]

        for i in range(config.get("FlowNetwork", "ntransforms")):
            transforms.append(MaskedAutoregressiveTransform(
                features=n_features,
                context=n_context,
                order=torch.randperm(features) if randperm else orders[i % 2],
                univariate=MonotonicRQSTransform,
                shapes=[(bins,), (bins,), (bins - 1,)],
                hidden_features=config.get("FlowNetwork", "hidden_features")
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
            ).to(device)
        
    elif config.get("FlowNetwork", "flow_model_type") == "zuko-cnf":
        model = zuko.flows.CNF(
            n_features, 
            context=n_context, 
            hidden_features=config.get("FlowNetwork", "hidden_features")
            ).to(device)

    elif config.get("FlowNetwork", "flow_model_type") == "zuko_nsf":
        model = zuko.flows.spline.NSF(
            n_features, 
            context=n_context, 
            transforms=config.get("FlowNetwork", "ntransforms"), 
            bins=config.get("FlowNetwork", "nsplines"), 
            hidden_features=config.get("FlowNetwork", "hidden_features")
            ).to(device)
        
    elif config.get("FlowNetwork", "flow_model_type") == "glasflow-nsf":
        model = glasflow.CouplingNSF(
            n_inputs=n_features,
            n_transforms=config.get("FlowNetwork", "ntransforms"),
            n_blocks_per_transform=len(config.get("FlowNetwork", "hidden_features")),
            n_conditional_inputs=n_context,
            n_neurons=config.get("FlowNetwork", "hidden_features")[0],
            num_bins=config.get("FlowNetwork", "nsplines")
        ).to(device)

    elif config.get("FlowNetwork", "flow_model_type") == "glasflow-enflow":
        # Not working yet
        model = glasflow.EnFlow(
            n_inputs=n_features,
            n_transforms=config.get("FlowNetwork", "n_transforms"),
            n_conditional_inputs=n_context,
            n_neurons=config.get("FlowNetwork", "hidden_features"),
            num_bins=config.get("FlowNetwork", "nsplines")
        ).to(device)
    else:
        print("-- No flow specified -- Using zuko nsf --")
        model = zuko.flows.spline.NSF(
            n_features, 
            context=n_context, 
            transforms=config.get("FlowNetwork", "ntransforms"), 
            bins=config.get("FlowNetwork", "nsplines"), 
            hidden_features=config.get("FlowNetwork", "hidden_features")
            ).to(device)

    return pre_model, model

def load_models(config, device=None):
    """Load in models from config

    Args:
        config (_type_): config dictionary
        device (_type_): which device to put the models on

    Returns:
        tuple: pre_model, model
    """

    if device is None:
        device = config.get("Training", "device")

    times, basis_dynamics, masses, strain, feature_shape, positions, all_dynamics, snr, basis_velocities = data_generation.generate_data(
        2, 
        config.get("Data", "basis_order"), 
        config.get("Data", "n_masses"), 
        config.get("Data", "sample_rate"), 
        n_dimensions=3, 
        detectors=config.get("Data", "detectors"), 
        window_strain=config.get("Data", "window_strain"), 
        window_acceleration=config.get("Data", "window_acceleration"),
        basis_type=config.get("Data", "basis_type"),
        data_type=config.get("Data", "data_type"),
        prior_args=config.get("Data", "prior_args"))

    n_basis = config.get("Data", "basis_order")
    if config.get("Data", "basis_type") == "fourier":
        n_basis += 2
    feature_shape = config.get("Data", "n_masses") + config.get("Data", "n_masses")*config.get("Data", "n_dimensions")*n_basis

    n_features = feature_shape#cshape*config.get("n_masses"]*config.get("n_dimensions"] + config.get("n_masses"]
    n_context = config.get("FlowNetwork", "n_context")
    n_input = config.get("Data", "sample_rate")*config.get("Data", "duration")

    pre_model, model = create_models(config, device)

    pre_model.to(device)
    model.to(device)
    
    weights = torch.load(os.path.join(config.get("General", "root_dir"),"test_model.pt"), map_location=device)

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

