import zuko
from zuko.transforms import (
    AffineTransform,
    MonotonicRQSTransform,
    RotationTransform,
    SigmoidTransform,
)
from zuko.flows import (
    Flow,
    GeneralCouplingTransform,
    MaskedAutoregressiveTransform,
    NeuralAutoregressiveTransform,
    Unconditional,
)
from zuko.distributions import DiagNormal
import numpy as np



def create_model(config, device):
    """create a convolutional to linear model with n_context outputs

    Args:
        conv_layers (_type_): convolutional layers [(input_channels, output_channels, filter_size, max_pool_size), (), ...]
        linear_layers (_type_): fully connected layers [layer1_size, layer2_size, ...]
        n_context (_type_): number of context inputs to flow (output size of this network)

    Returns:
        _type_: _description_
    """

    times, labels, strain, cshape, positions = generate_data(2, config["chebyshev_order"], config["n_masses"], config["sample_rate"], n_dimensions=config["n_dimensions"], detectors=config["detectors"], window=config["window"], return_windowed_coeffs=config["return_windowed_coeffs"])

    n_features = cshape*config["n_masses"]*config["n_dimensions"] + config["n_masses"]
    n_context = config["sample_rate"]*2

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

    if config["custom_flow"] == "test":
        bins = config["nsplines"]
        randperm = False
        orders = [
            torch.arange(n_features),
            torch.flipud(torch.arange(n_features)),
        ]
        transforms = [
            MaskedAutoregressiveTransform(
                features=n_features,
                context=n_context,
                order=torch.randperm(features) if randperm else orders[i % 2],
                univariate=MonotonicRQSTransform,
                shapes=[(bins,), (bins,), (bins - 1,)],
                hidden_features=config["hidden_features"]
            )
            for i in range(config["ntransforms"])
        ]

        transforms.append(Unconditional(lambda: SigmoidTransform().inv))  # y = logit(x)
        
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
    times, labels, strain, cshape, positions = generate_data(2, config["chebyshev_order"], config["n_masses"], config["sample_rate"], n_dimensions=config["n_dimensions"], detectors=config["detectors"], window=config["window"], return_windowed_coeffs=config["return_windowed_coeffs"])

    n_features = cshape*config["n_masses"]*config["n_dimensions"] + config["n_masses"]
    n_context = config["sample_rate"]*2

    pre_model, model = create_model(config, device)

    pre_model.to(device)
    model.to(device)
    
    weights = torch.load(os.path.join(config["root_dir"],"test_model.pt"), map_location=device)

    pre_model.load_state_dict(weights["pre_model_state_dict"])

    model.load_state_dict(weights["model_state_dict"])

    return pre_model, model


def get_dynamics(coeffmass_samples, times, n_masses, chebyshev_order, n_dimensions, poly_type="chebyshev"):
    """get the dynamics of the system from polynomial cooefficients and masses

    Args:
        coeffmass_samples (_type_): samples of the coefficients and masses
        times (_type_): times when to evaluate the polynomial
        n_masses (_type_): how many masses 
        chebyshev_order (_type_): order of the polynomimal
        n_dimensions (_type_): how many dimensions 

    Returns:
        tuple: (coefficients, masses, timeseries)
    """
    #print("msshape", np.shape(coeffmass_samples))
    masses = coeffmass_samples[-n_masses:]
    coeffs = coeffmass_samples[:-n_masses].reshape(n_masses,chebyshev_order, n_dimensions)

    tseries = np.zeros((n_masses, n_dimensions, len(times)))
    for mass_index in range(n_masses):
        for dim_index in range(n_dimensions):
            tseries[mass_index, dim_index] = polynomial_dict[poly_type]["val"](times, coeffs[mass_index, :, dim_index])

    return coeffs, masses, tseries

def get_strain_from_samples(
    times, 
    recon_masses, 
    source_masses,
    recon_coeffs, 
    source_coeffs, 
    detectors=["H1"],
    return_windowed_coeffs=False, 
    window=False, 
    poly_type="chebyshev"):
    """_summary_

    Args:
        times (_type_): _description_
        recon_masses (_type_): _description_
        source_masses (_type_): _description_
        recon_coeffs (_type_): _description_
        source_coeffs (_type_): _description_
        detectors (list, optional): _description_. Defaults to ["H1"].
        return_windowed_coeffs (bool, optional): _description_. Defaults to False.
        window (bool, optional): _description_. Defaults to False.
        poly_type (str, optional): _description_. Defaults to "chebyshev".

    Returns:
        _type_: _description_
    """
    # if there is a window and I and not predicting the windowed coefficients
    n_masses, n_coeffs, n_dimensions = np.shape(recon_coeffs)
    if not return_windowed_coeffs and window != False:
        n_recon_coeffs = []
        n_source_coeffs = []
        # for each mass perform the window on the xyz positions (acceleration)
        for mass in range(n_masses):
            temp_recon, win_coeffs = perform_window(times, recon_coeffs[mass], window, poly_type=poly_type)
            n_recon_coeffs.append(temp_recon)
            if source_coeffs is not None:
                temp_source, win_coeffs = perform_window(times, source_coeffs[mass], window, poly_type=poly_type)
                n_source_coeffs.append(temp_source)
            

        
        # update the coefficients with the windowed version
        recon_coeffs = np.array(n_recon_coeffs)
        if source_coeffs is not None:
            source_coeffs = np.array(n_source_coeffs)

    recon_strain_coeffs = compute_hTT_coeffs(recon_masses, np.transpose(recon_coeffs, (0,2,1)), poly_type=poly_type)
    if source_coeffs is not None:
        source_strain_coeffs = compute_hTT_coeffs(source_masses, np.transpose(source_coeffs, (0,2,1)), poly_type=poly_type)

    recon_energy = compute_energy_loss(times, recon_masses, np.transpose(recon_coeffs, (0,2,1)), poly_type=poly_type)
    source_energy = []
    if source_coeffs is not None:
        source_energy = compute_energy_loss(times, source_masses, np.transpose(source_coeffs, (0,2,1)), poly_type=poly_type)

    recon_strain = []
    source_strain = []
    for detector in detectors:
        recon_strain.append(compute_strain_from_coeffs(times, recon_strain_coeffs, detector=detector, poly_type=poly_type))
        if source_coeffs is not None:
            source_strain.append(compute_strain_from_coeffs(times, source_strain_coeffs, detector=detector, poly_type=poly_type))

    return recon_strain, source_strain, recon_energy, source_energy, recon_coeffs, source_coeffs
