import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from data_generation import generate_data, compute_hTT_coeffs, compute_strain_from_coeffs, perform_window, polynomial_dict, compute_energy_loss
import data_generation
import make_animations as animations
import plotting
from train_model import get_dynamics, create_model, load_models, get_strain_from_samples
import zuko
import argparse
import copy


def chirp_positions(times, m1, m2, tc, detectors=["H1", "L1", "V1"], chebyshev_order=10, window=False, poly_type="chebyshev", root_dir="./"):
    """ Get inspiral signal
    """
    #chebyshev_order = 30
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
    
    #m1pos = np.vstack([-r1*np.cos(theta), np.zeros(np.shape(theta)), -r1*np.sin(theta)])
    #m2pos = np.vstack([r2*np.cos(theta), np.zeros(np.shape(theta)), r2*np.sin(theta)])

    #m1pos = np.vstack([np.zeros(np.shape(theta)), -r1*np.cos(theta), -r1*np.sin(theta)])
    #m2pos = np.vstack([np.zeros(np.shape(theta)), r2*np.cos(theta), r2*np.sin(theta)])

    positions = np.array([m1pos,m2pos])

    positions = positions/np.max(positions)

    norm_masses = np.array([m1, m2])/np.sum([m1,m2])

    chebyorder = chebyshev_order
    if not window:
        cheb_dynamics = np.zeros((2, 3, chebyorder))
    else:
        cheb_dynamics = []

    for mind in range(2):
        temp_dyn = np.zeros((chebyorder, 3))
        for dimind in range(3):
            temp_dyn[:,dimind] = polynomial_dict[poly_type]["fit"](
                times, 
                positions[mind, dimind], 
                chebyorder-1)

        if window:
            temp_dyn, win_coeffs = perform_window(
                times, 
                temp_dyn, 
                window, 
                order=chebyorder, 
                poly_type=poly_type)
            cheb_dynamics.append(temp_dyn.T)
        else:
            cheb_dynamics[mind] = temp_dyn.T


    cheb_dynamics = np.array(cheb_dynamics)
    #coeffs = np.array([data_generation.generate_random_coefficients(chebyorder, 3) for _ in range(2)])
    #print(np.shape(coeffs))
    print("masses",norm_masses)

    # change to mass, dim, order
    #cheb_dynamics = np.transpose(coeffs, (0,2,1))

    timeseries_dynamics = np.zeros((2,3,len(times)))
    for i in range(2):
        for j in range(3):
            timeseries_dynamics[i,j] = polynomial_dict[poly_type]["val"](times, cheb_dynamics[i,j])

    fig, ax = plt.subplots(nrows=3)
    ax[0].plot(times, timeseries_dynamics[0,0])
    ax[1].plot(times, timeseries_dynamics[0,1])
    ax[2].plot(times, timeseries_dynamics[0,2])
    fig.savefig(os.path.join(root_dir, "chirp_positions.png"))



    print("maxmin cheby", np.max(cheb_dynamics), np.min(cheb_dynamics))
    cheb_dynamics = np.array(cheb_dynamics)
    n_chebyorder = np.shape(cheb_dynamics)[-1]

    print("chebshape", np.shape(cheb_dynamics))
    # should be n_masses, n_dim, n_coeffs
    #np.transpose(cheb_dynamics, (0,2,1))
    strain_coeffs = compute_hTT_coeffs(norm_masses, cheb_dynamics, poly_type=poly_type)

    energy = compute_energy_loss(times, norm_masses, cheb_dynamics, poly_type=poly_type)

    """
    print(sm1, sm2)
    all_dynamics = sm1*cheb_dynamics[0] + sm2*cheb_dynamics[1]

    temp_strain_timeseries = generate_3d_derivative(all_dynamics.T.reshape(n_chebyorder, 3), times)
    """

    strain_timeseries = np.zeros((len(detectors), len(times)))
    for dind, detector in enumerate(detectors):
        strain = compute_strain_from_coeffs(times, strain_coeffs, detector=detector, poly_type=poly_type)
        #strain = compute_strain(temp_strain_timeseries, detector, poly_type=poly_type)
        strain_timeseries[dind] = strain

    fig, ax = plt.subplots(nrows=3)
    ax[0].plot(times, strain_timeseries[0])
    ax[1].plot(times, strain_timeseries[1])
    ax[2].plot(times, strain_timeseries[2])
    fig.savefig(os.path.join(root_dir, "chirp_strain.png"))


    return positions, cheb_dynamics, timeseries_dynamics, strain_timeseries, energy

def run_chirp_test(config, mass1=5000, mass2=5000):

    
    plot_out = os.path.join(config["root_dir"], f"test_chirp_m1-{mass1}_m2-{mass2}_edge2")
    if not os.path.isdir(plot_out):
        os.makedirs(plot_out)

    poly_type = "chebyshev"
    
    pre_model, model = load_models(config, device="cpu")
    
    times = np.linspace(-1,1,config["sample_rate"])

    chebyshev_order = config["chebyshev_order"]
    #m1,m2 = 2000,500
    m1,m2 = mass1, mass2
    dynamics, cheb_dynamics, all_dynamics, data, energy = chirp_positions(times, m1, m2, 1.1, detectors=config["detectors"], chebyshev_order=chebyshev_order, window=config["window"], root_dir=plot_out)
    

    n_masses = 2
    n_dimensions = 3
   
    animations.make_3d_animation(plot_out, 100, all_dynamics, 0.01*np.array([m1,m2]), None, None)

    
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

    fig, ax = plt.subplots(nrows=2)
    fct = 1
    ax[0].plot(times[fct:-fct], data[0][fct:-fct])
    ax[1].plot(times, energy)
    fig.savefig(os.path.join(plot_out,"test_chirp.png"))

    source_tseries = all_dynamics
    source_masses = np.array([m1,m2])/np.sum([m1, m2])
    batch = 0

    input_data = pre_model(torch.from_numpy(np.array([data])).to(torch.float32))

    nsamples = 200
    n_animate_samples = 50
    multi_coeffmass_samples = model(input_data).sample((nsamples, )).cpu().numpy()

    m_recon_tseries, m_recon_masses = np.zeros((nsamples, n_masses, n_dimensions, len(times))), np.zeros((nsamples, n_masses))
    m_recon_strain = np.zeros((nsamples, 3, len(times)))
    for i in range(nsamples):
        #print(np.shape(multi_coeffmass_samples[i]))
        t_co, t_mass, t_time = get_dynamics(multi_coeffmass_samples[i][0], times, n_masses, chebyshev_order, n_dimensions, poly_type=poly_type)
        m_recon_tseries[i] = t_time
        m_recon_masses[i] = t_mass

        recon_strain, recon_energy, source_strain, source_energy, m_recon_coeffs, source_coeffs = get_strain_from_samples(
            times, 
            t_mass,
            None,
            t_co, 
            None, 
            detectors=["H1","L1","V1"],
            return_windowed_coeffs=config["return_windowed_coeffs"], 
            window=config["window"], 
            poly_type=config["poly_type"])

        m_recon_strain[i] = recon_strain

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


        if source_masses[0] - source_masses[1] < 0:
            new_source_tseries = copy.copy(source_tseries)
            new_source_tseries[0] = source_tseries[1]
            new_source_tseries[1] = source_tseries[0]

            new_source_masses = copy.copy(source_masses)
            new_source_masses[0] = source_masses[1]
            new_source_masses[1] = source_masses[0]

            source_masses = new_source_masses
            source_tseries = new_source_tseries

    plotting.plot_sampled_reconstructions(
                times, 
                config["detectors"], 
                m_recon_strain, 
                data, 
                fname = os.path.join(plot_out,f"recon_strain_dist_{batch}.png"))

    plotting.plot_sample_separations(
                times, 
                source_tseries, 
                m_recon_tseries, 
                fname=os.path.join(plot_out,f"separations_{batch}.png"))


    plotting.plot_mass_distributions(
                m_recon_masses,
                source_masses,
                fname=os.path.join(plot_out,f"massdistributions_{batch}.png"))

            
    animations.line_of_sight_animation(
                m_recon_tseries, 
                m_recon_masses, 
                source_tseries, 
                source_masses, 
                os.path.join(plot_out,f"2d_massdist_{batch}.gif"))

    print("source", np.shape(source_tseries), np.shape(source_masses))

    animations.make_3d_distribution(
                plot_out, 
                batch, 
                m_recon_tseries[:n_animate_samples], 
                m_recon_masses[:n_animate_samples], 
                source_tseries, 
                source_masses)

    animations.make_3d_distribution_zproj(
                plot_out, 
                batch, 
                m_recon_tseries, 
                m_recon_masses, 
                source_tseries, 
                source_masses)



    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=False, default="none")
    parser.add_argument("-m1", "--mass2", type=float, required=False, default=5000)
    parser.add_argument("-m2", "--mass1", type=float, required=False, default=5000)
    args = parser.parse_args()


    with open(os.path.abspath(args.config), "r") as f:
        config = json.load(f)

    run_chirp_test(config, args.mass1, args.mass2)