import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from massdynamics.data_generation import (
    data_generation,
    data_processing,
    compute_waveform
)
from massdynamics.plotting import plotting, make_animations
from massdynamics.create_model import (
    create_models,
    load_models, 
)
from massdynamics.basis_functions import basis
from massdynamics import window_functions
import zuko
import argparse
import copy
import sys
from pycbc.waveform import get_td_waveform

def fit_positions_with_polynomial(
    times, 
    positions, 
    basis_order=8, 
    window="none", 
    basis_type="chebyshev"):
    """fit the 3d positions with a polynomial

    Args:
        times (_type_): _description_
        positions (_type_): _description_
        basis_order (int, optional): _description_. Defaults to 8.
        window (bool, optional): _description_. Defaults to False.
        basis_type (str, optional): _description_. Defaults to "chebyshev".

    Returns:
        _type_: _description_
    """
  
    n_masses, n_dimensions, n_cheby = np.shape(positions)
    dtype = np.float64
    if basis_type == "fourier":
        basis_order = int(basis_order/2 + 1)
        dtype = complex

    if window == "none" or not window:
        basis_dynamics = np.zeros((n_masses, n_dimensions, basis_order), dtype=dtype)
    else:
        basis_dynamics = []

    for mind in range(n_masses):
        # this way around as fit to second last dimensions
        temp_dyn = np.zeros((n_dimensions, basis_order), dtype=dtype)
        dimfit = basis[basis_type]["fit"](
            times, 
            positions[mind], 
            basis_order)

        temp_dyn[:] = dimfit
            
        if window != "none":
            temp_dyn2, win_coeffs = window_functions.perform_window(
                times, 
                temp_dyn.T, 
                window, 
                order=(basis_order-1)*2, 
                basis_type=basis_type)
            basis_dynamics.append(temp_dyn2.T)
        else:
            basis_dynamics[mind] = temp_dyn


    basis_dynamics = np.array(basis_dynamics, dtype=dtype)

    return basis_dynamics

def run_chirp_test(config, mass1=500, mass2=500, rotate_angle=0.0):

    
    plot_out = os.path.join(config["root_dir"], f"test_imrphenom_m1-{mass1}_m2-{mass2}_rotate{rotate_angle}")

    if not os.path.isdir(plot_out):
        os.makedirs(plot_out)

    basis_type = config["basis_type"]
    
    pre_model, model = load_models(config, device="cpu")
    
    times = np.linspace(0,1,config["sample_rate"])
    upsample_times = np.linspace(0,1,config["plot_sample_rate"])

    basis_order = config["basis_order"]

    srate = config["sample_rate"]
    n_masses=2
    n_dimensions=3
    batch=0

    hp, hc = get_td_waveform(approximant="IMRPhenomPv2",
                        mass1 = mass1,
                        mass2 = mass2,
                        delta_t = 1./srate,
                        f_lower=5
                        )
    
    source_masses = np.array([0.5,0.5])
    
    det_data = []
    for det in config["detectors"]:
        strain = compute_waveform.compute_strain(np.array([[hp,hc]]), detector=det)[10*srate:11*srate]
        det_data.append(strain)

    data = np.array(det_data) / (0.5*np.max(det_data))

    print("datashape", np.shape(data))
    freq_strain = basis[config["basis_type"]]["fit"](times, data, basis_order)
    source_strain = basis[config["basis_type"]]["val"](upsample_times, freq_strain)



    norm_data, norm_factor = data_processing.normalise_data(data, pre_model.norm_factor)    
    print("normfactor", norm_factor, np.max(data), np.max(norm_data))

    input_data = pre_model(torch.from_numpy(np.array([norm_data])).to(torch.float32))

    nsamples = 100
    n_animate_samples = 100
    multi_coeffmass_samples = model(input_data).sample((nsamples, )).cpu().numpy()

    multi_coeffmass_samples, nf, mf = data_processing.unnormalise_labels(
                multi_coeffmass_samples[:,0], 
                pre_model.label_norm_factor, 
                pre_model.mass_norm_factor,
                n_masses=n_masses)

    m_recon_tseries, m_recon_masses = np.zeros((nsamples, n_masses, n_dimensions, len(upsample_times))), np.zeros((nsamples, n_masses))
    m_recon_strain = np.zeros((nsamples, 3, len(upsample_times)))

    multi_mass_samples, multi_coeff_samples = data_processing.samples_to_positions_masses(
                multi_coeffmass_samples, 
                n_masses,
                basis_order,
                n_dimensions,
                basis_type=basis_type)

    for i in range(nsamples):
        t_co = multi_coeff_samples[i]
        t_mass = multi_mass_samples[i]
        t_time = compute_waveform.get_time_dynamics(
            multi_coeff_samples[i],
            upsample_times,  
            basis_type=basis_type)

        
        m_recon_tseries[i] = t_time
        m_recon_masses[i] = t_mass


        temp_recon_strain, temp_recon_energy, temp_m_recon_coeffs = data_processing.get_strain_from_samples(
            upsample_times, 
            t_mass,
            np.array(t_co), 
            detectors=["H1","L1","V1"],
            return_windowed_coeffs=config["return_windowed_coeffs"], 
            window=config["window"], 
            basis_type=config["basis_type"])

        #temp_recon_strain, _ = data_processing.unnormalise_data(temp_recon_strain, pre_model.norm_factor)

        m_recon_strain[i] = temp_recon_strain

    if n_masses == 2:
        neginds = m_recon_masses[:,0] - m_recon_masses[:,1] < 0

        new_recon_tseries = copy.copy(m_recon_tseries)
        new_recon_tseries[neginds, 0] = m_recon_tseries[neginds, 1]
        new_recon_tseries[neginds, 1] = m_recon_tseries[neginds, 0]

        new_recon_masses = copy.copy(m_recon_masses)
        new_recon_masses[neginds, 0] = m_recon_masses[neginds, 1]
        new_recon_masses[neginds, 1] = m_recon_masses[neginds, 0]

        m_recon_masses = new_recon_masses
        m_recon_tseries = new_recon_tseries


    print(np.shape(upsample_times), np.shape(m_recon_strain), np.shape(norm_data))
    plotting.plot_sampled_reconstructions(
                upsample_times, 
                config["detectors"], 
                m_recon_strain, 
                source_strain, 
                fname = os.path.join(plot_out,f"recon_strain_dist_{batch}.png"))


    plotting.plot_mass_distributions(
                m_recon_masses,
                source_masses,
                fname=os.path.join(plot_out,f"massdistributions_{batch}.png"))

    print(np.max(m_recon_tseries), np.min(m_recon_tseries))
    make_animations.line_of_sight_animation(
                m_recon_tseries, 
                m_recon_masses, 
                None, 
                source_masses, 
                os.path.join(plot_out,f"2d_massdist_{batch}.gif"))


    
    make_animations.make_3d_distribution(
                plot_out, 
                m_recon_tseries[:n_animate_samples], 
                m_recon_masses[:n_animate_samples], 
                None, 
                source_masses,
                fname = os.path.join(plot_out,f"3d_distribution_{batch}.png"))

    make_animations.heatmap_projections(
                m_recon_tseries, 
                m_recon_masses, 
                None, 
                source_masses, 
                os.path.join(plot_out,f"heatmap_projections_{batch}.gif"),
                duration=5)

    make_animations.make_distribution_projections(
                plot_out, 
                batch, 
                m_recon_tseries, 
                m_recon_masses, 
                None, 
                source_masses,
                strain=m_recon_strain,
                true_strain=source_strain,
                duration=5)



    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=False, default="none")
    parser.add_argument("-m1", "--mass2", type=float, required=False, default=5000)
    parser.add_argument("-m2", "--mass1", type=float, required=False, default=5000)
    args = parser.parse_args()


    with open(os.path.abspath(args.config), "r") as f:
        config = json.load(f)


    run_chirp_test(config, args.mass1, args.mass2, rotate_angle=0.0)