import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from data_generation import generate_data, generate_3d_derivative, compute_strain, perform_window
from make_animations import make_3d_animation, make_3d_distribution
from train_model import get_dynamics, create_model
import zuko

def load_models(config):

    times, labels, strain, cshape, positions = generate_data(2, config["chebyshev_order"], config["n_masses"], config["sample_rate"], n_dimensions=config["n_dimensions"], detectors=config["detectors"], window=config["window"], return_windowed_coeffs=config["return_windowed_coeffs"])

    n_features = cshape*config["n_masses"]*config["n_dimensions"] + config["n_masses"]
    n_context = config["sample_rate"]*2

    pre_model = create_model(config["conv_layers"], config["linear_layers"], n_context).to("cpu")

    model = zuko.flows.spline.NSF(n_features, context=n_context, transforms=config["ntransforms"], bins=config["nsplines"], hidden_features=config["hidden_features"]).to("cpu")
    
    weights = torch.load(os.path.join(config["root_dir"],"test_model.pt"), map_location="cpu")

    pre_model.load_state_dict(weights["pre_model_state_dict"])

    model.load_state_dict(weights["model_state_dict"])

    return pre_model, model

def chirp_positions(times, m1, m2, tc, detectors=["H1", "L1", "V1"], chebyshev_order=10, window=False):

    G = 6.6e-11
    c = 3e8
    M_sun = 1.9e30
    r_s = 2*G*M_sun/c**2
    # chirp mass
    Mc = ((m1*m2)**(3./5.) )/(m1 + m2)**(1./5.)
    # orbital frequency
    f = 0.5*( (8*np.pi)**(8./3.)/5. * (G*Mc*M_sun/c**3)**(5./3.) *(tc-times) )**(-3./8.) 
    # orbital separation
    r = (( G * M_sun*(m1 + m2)/(2*np.pi*f)**2 )**(1./3.))/r_s 

    sm1 = m1/(m1+m2)
    sm2 = m2/(m1+m2)
    r1 = r*sm2
    r2 = r*sm1
    theta = 2*np.pi*f*times
    m1pos = np.vstack([-r1*np.cos(theta), -r1*np.sin(theta), np.zeros(np.shape(theta))])
    m2pos = np.vstack([r2*np.cos(theta), r2*np.sin(theta), np.zeros(np.shape(theta))])

    positions = np.array([m1pos,m2pos])
    chebyorder = chebyshev_order
    if not window:
        cheb_dynamics = np.zeros((2, 3, chebyorder))
    else:
        cheb_dynamics = []
    for mind in range(2):
        temp_dyn = np.zeros((chebyorder, 3))
        for dimind in range(3):
            temp_dyn[:,dimind] = np.polynomial.chebyshev.chebfit(times, positions[mind, dimind], chebyorder-1)
        if window:
            temp_dyn, win_coeffs = perform_window(times, temp_dyn, window, order=chebyorder)
            cheb_dynamics.append(temp_dyn.T)
        else:
            cheb_dynamics[mind] = temp_dyn.T
            

    
    cheb_dynamics = np.array(cheb_dynamics)
    n_chebyorder = np.shape(cheb_dynamics)[-1]
    print(sm1, sm2)
    all_dynamics = sm1*cheb_dynamics[0] + sm2*cheb_dynamics[1]

    temp_strain_timeseries = generate_3d_derivative(all_dynamics.T.reshape(n_chebyorder, 3), times)

    strain_timeseries = np.zeros((len(detectors), len(times)))
    for dind, detector in enumerate(detectors):
        strain = compute_strain(temp_strain_timeseries, detector)
        strain_timeseries[dind] = strain



    return positions, cheb_dynamics, all_dynamics, strain_timeseries

def run_chirp_test(config):

    
    plot_out = os.path.join(config["root_dir"], "test_chirp")
    if not os.path.isdir(plot_out):
        os.makedirs(plot_out)
    """
    pre_model, model = load_models(config)
    """
    times = np.linspace(-1,1,config["sample_rate"])

    chebyshev_order = config["chebyshev_order"]
    m1,m2 = 10000,9000
    dynamics, cheb_dynamics, all_dynamics, data = chirp_positions(times, m1, m2, 1.1, detectors=config["detectors"], chebyshev_order=chebyshev_order, window=config["window"])
    

    n_masses = 2
    n_dimensions = 3
   
    make_3d_animation(plot_out, 0, dynamics, 0.01*np.array([m1,m2]), None, None)

    
    reconstructx = np.polynomial.chebyshev.chebval(times, cheb_dynamics[0][0])
    reconstructy = np.polynomial.chebyshev.chebval(times, cheb_dynamics[0][1])
    reconstructz = np.polynomial.chebyshev.chebval(times, cheb_dynamics[0][2])

    fig, ax = plt.subplots(nrows=3)
    ax[0].plot(dynamics[0,0])
    ax[0].plot(reconstructx)
    ax[1].plot(dynamics[0,1])
    ax[1].plot(reconstructy)
    ax[2].plot(dynamics[0,2])
    ax[2].plot(reconstructz)
    fig.savefig(os.path.join(plot_out, "test_move.png"))

    fig, ax = plt.subplots()
    fct = 1
    ax.plot(times[fct:-fct], data[0][fct:-fct])
    fig.savefig(os.path.join(plot_out,"test_chirp.png"))

    """

    input_data = pre_model(torch.from_numpy(np.array([data])).to(torch.float32))
    coeffmass_samples = model(input_data).sample().cpu().numpy()

    recon_coeffs, recon_masses, recon_tseries = get_dynamics(coeffmass_samples[0], times, n_masses, chebyshev_order, n_dimensions)

    fig, ax = plt.subplots(nrows = 4)

    recon_weighted_coeffs = np.sum(recon_coeffs * recon_masses[:, None, None], axis=0)

    recon_strain_tensor = generate_3d_derivative(recon_weighted_coeffs, times)

    recon_strain = []
    for detector in config["detectors"]:
        recon_strain.append(compute_strain(recon_strain_tensor, detector=detector))

    for i in range(len(config["detectors"])):
        print(np.shape(times), np.shape(recon_strain))

        ax[i].plot(times, recon_strain[i], label="recon")
        ax[i].plot(times, data[i], label="source")

    fig.savefig(os.path.join(plot_out, f"reconstructed_{0}.png"))


    nsamples = 50
    multi_coeffmass_samples = model(input_data).sample((nsamples, )).cpu().numpy()

    #print("multishape", multi_coeffmass_samples.shape)
    m_recon_tseries, m_recon_masses = np.zeros((nsamples, n_masses, n_dimensions, len(times))), np.zeros((nsamples, n_masses))
    for i in range(nsamples):
        #print(np.shape(multi_coeffmass_samples[i]))
        t_co, t_mass, t_time = get_dynamics(multi_coeffmass_samples[i][0], times, n_masses, chebyshev_order, n_dimensions)
        m_recon_tseries[i] = t_time
        m_recon_masses[i] = t_mass

    make_3d_distribution(plot_out, 0, m_recon_tseries, m_recon_masses, None, None)

    """

if __name__ == "__main__":

    root_dir = "/home/joseph.bayley/projects/mass_dynamics_reconstruction/test_cheb8_3d_3det_hannwindowpost"
    
    with open(os.path.join(root_dir, "config.json"), "r") as f:
        config = json.load(f)

    run_chirp_test(config)