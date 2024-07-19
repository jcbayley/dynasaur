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

def generate_m1m2_pos(times, m1, m2, tc, orientation="xy"):
    """Generate the positions of the inspiral to first order

    Args:
        times (array): times at which to evaluate the signal
        m1 (float): mass of object 1
        m2 (float): mass of object 2
        tc (float): time of coalescence

    Returns:
        array: array of the normalised masses
        array: array of the positions of each mass
    """
    #basis_order = 30
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

    if orientation == "xy":
        m1pos = np.vstack([-r1*np.cos(theta), -r1*np.sin(theta), np.zeros(np.shape(theta))])
        m2pos = np.vstack([r2*np.cos(theta), r2*np.sin(theta), np.zeros(np.shape(theta))])
    elif orientation == "xz":
        m1pos = np.vstack([-r1*np.cos(theta), np.zeros(np.shape(theta)), -r1*np.sin(theta)])
        m2pos = np.vstack([r2*np.cos(theta), np.zeros(np.shape(theta)), r2*np.sin(theta)])
    elif orientation == "yz":
        m1pos = np.vstack([np.zeros(np.shape(theta)), -r1*np.cos(theta), -r1*np.sin(theta)])
        m2pos = np.vstack([np.zeros(np.shape(theta)), r2*np.cos(theta), r2*np.sin(theta)])
    else:
        raise Exception(f"No orientation {orientation}")


    positions = np.array([m1pos,m2pos])

    positions = positions/np.max(positions)

    masses = np.array([m1,m2])
    norm_masses = masses/np.sum(masses)

    return norm_masses, positions, masses

def generate_m1m2_pos_1d(times, m1, m2, tc, orientation="xy"):
    """Generate the positions of the inspiral to first order

    Args:
        times (array): times at which to evaluate the signal
        m1 (float): mass of object 1
        m2 (float): mass of object 2
        tc (float): time of coalescence

    Returns:
        array: array of the normalised masses
        array: array of the positions of each mass
    """
    #basis_order = 30

    ph_offset = 0
    x = np.sin(2*np.pi*times*1 + ph_offset)
    y = np.sin(2*np.pi*times*1 + ph_offset)
    z = np.zeros(len(times))


    if orientation == "xy":
        m1pos = np.vstack([x,y,z])
        #m2pos = np.vstack([x,y,z])
    elif orientation == "yx":
        m1pos = np.vstack([x,-y,z])
        #m2pos = np.vstack([x,y,z])
    elif orientation == "yy":
        m1pos = np.vstack([-x,-y,z])
        #m2pos = np.vstack([x,y,z])
    elif orientation == "yz":
        m1pos = np.vstack([x,z,y])
        #m2pos = np.vstack([x,y,z])
    elif orientation == "offz":
        m1pos = np.vstack([x,y,z+0.2])
        #m2pos = np.vstack([x,y,z])
    else:
        raise Exception(f"No orientation {orientation}")
    """
    elif orientation == "xz":
        m1pos = np.vstack([x,z,y])
        #m1pos = np.vstack([-r1*np.cos(theta), np.zeros(np.shape(theta)), -r1*np.sin(theta)])
        #m2pos = np.vstack([r2*np.cos(theta), np.zeros(np.shape(theta)), r2*np.sin(theta)])
    elif orientation == "yz":
        m1pos = np.vstack([z,y,x])
        #m1pos = np.vstack([np.zeros(np.shape(theta)), -r1*np.cos(theta), -r1*np.sin(theta)])
        #m2pos = np.vstack([np.zeros(np.shape(theta)), r2*np.cos(theta), r2*np.sin(theta)])
    """
    

    positions = np.array([m1pos,])

    positions = 0.5*positions/np.max(positions)

    norm_masses = np.array([m1,])/np.sum([m1,])

    return norm_masses, positions

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

"""
def get_ts_dynamics(times, basis_dynamics, basis_type="chebyshev"):

    n_masses, n_dimensions, n_coeffs = np.shape(basis_dynamics)
    timeseries_dynamics = np.zeros((n_masses,n_dimensions,len(times)))
    for i in range(n_masses):
        for j in range(n_dimensions):
            timeseries_dynamics[i,j] = basis[basis_type]["val"](times, basis_dynamics[i,j])
        
    return timeseries_dynamics

def get_waveform(times, norm_masses, basis_dynamics, detectors, basis_type="chebyshev"):

    strain_coeffs = compute_hTT_coeffs(norm_masses, basis_dynamics, basis_type=basis_type)

    energy = compute_energy_loss(times, norm_masses, basis_dynamics, basis_type=basis_type)

    strain_timeseries = np.zeros((len(detectors), len(times)))
    for dind, detector in enumerate(detectors):
        strain = compute_strain_from_coeffs(times, strain_coeffs, detector=detector, basis_type=basis_type)
        #strain = compute_strain(temp_strain_timeseries, detector, basis_type=basis_type)
        strain_timeseries[dind] = strain
    
    return strain_timeseries, energy
"""
def test_different_orientations(times, m1, m2, tc, basis_order, detectors, window="none", basis_type="chebyshev", root_dir="./"):

    orientations = ["xy", "yx", "yy", "offz"]
    positions = {}
    strain_timeseries = {}
    energy = {}
    basis_dynamics = {}
    norm_masses = {}
    dynamics = {}
    for orient in orientations:
           
        norm_masses[orient], positions[orient] = generate_m1m2_pos_1d(
            times, 
            m1, 
            m2, 
            tc, 
            orientation=orient)

        
        basis_dynamics[orient] = fit_positions_with_polynomial(
            times, 
            positions[orient], 
            basis_order=basis_order, 
            window=window, 
            basis_type=basis_type)
        
        dynamics[orient] = compute_waveform.get_time_dynamics(
            times, 
            basis_dynamics[orient], 
            basis_type=basis_type)
        
        print(np.shape(basis_dynamics[orient]))

        strain_timeseries[orient], energy[orient] = compute_waveform.get_waveform(
            times, 
            norm_masses[orient], 
            basis_dynamics[orient], 
            detectors, 
            basis_type=basis_type)

    fig, ax = plt.subplots(nrows = len(detectors) + 1)
    
    for orient in orientations:
        for detind in range(len(detectors)):
            ax[detind].plot(strain_timeseries[orient][detind])

    fig.savefig(os.path.join(root_dir, "orient_strain.png"))

    fig, ax = plt.subplots(nrows = 3, ncols=len(orientations))
    sind = 0
    for i,orient in enumerate(orientations):
        print(np.shape(dynamics[orient]))
        print(np.shape(positions[orient]))
        for dimind in range(3):
            ax[dimind, i].plot(dynamics[orient][0][dimind], color=f"C{sind}")
            ax[dimind, i].plot(positions[orient][0][dimind], color="k")
            sind += 1

    fig.savefig(os.path.join(root_dir, "orient_positions.png"))

    m_recon_tseries = np.array([
        dynamics[orient] for orient in orientations
    ])
    m_recon_masses = np.array([
        norm_masses[orient] for orient in orientations
    ])

    make_animations.make_3d_distribution(
                root_dir, 
                0, 
                m_recon_tseries, 
                m_recon_masses, 
                None, 
                None)
    
def test_1and2_masses(times, m1, m2, tc, basis_order, detectors, window="none", basis_type="chebyshev", root_dir="./"):

    orientations = ["twomass", "onemass", "othermass"]
    positions = {}
    strain_timeseries = {}
    energy = {}
    basis_dynamics = {}
    norm_masses = {}
    dynamics = {}
    for orient in orientations:
           
        norm_masses[orient], positions[orient] = generate_m1m2_pos(
            times, 
            m1, 
            m2, 
            tc, 
            orientation="xy")

        if orient == "onemass":
            positions[orient] = positions[orient][:1,:]
            norm_masses[orient] = norm_masses[orient][:1]    
        elif orient == "othermass":
            positions[orient] = positions[orient][1:, :]
            norm_masses[orient] = norm_masses[orient][1:]  

        print("shpo", np.shape(positions[orient]))
        basis_dynamics[orient] = fit_positions_with_polynomial(
            times, 
            positions[orient], 
            basis_order=basis_order, 
            window=window, 
            basis_type=basis_type)
        
        dynamics[orient] = get_ts_dynamics(
            times, 
            basis_dynamics[orient], 
            basis_type=basis_type)

        strain_timeseries[orient], energy[orient] = compute_waveform.get_waveform(
            times, 
            norm_masses[orient], 
            basis_dynamics[orient], 
            detectors, 
            basis_type=basis_type)

    fig, ax = plt.subplots(nrows = len(detectors) + 1)
    
    lss = ["-", "--", "-."]
    for oind,orient in enumerate(orientations):
        for detind in range(len(detectors)):
            ax[detind].plot(strain_timeseries[orient][detind], ls=lss[oind])

    smmed = np.sum([strain_timeseries[orient][0] for orient in ["onemass", "othermass"]], axis=0)
    print(np.shape(smmed))
    ax[3].plot(times, smmed)
    ax[3].plot(times, strain_timeseries["twomass"][0], ls="--")
    fig.savefig(os.path.join(root_dir, "orient_strain_1.png"))

    fig, ax = plt.subplots(nrows = 3, ncols=len(orientations))
    sind = 0
    for i,orient in enumerate(orientations):
        print("dynshape", np.shape(dynamics[orient]))
        print("posshape", np.shape(positions[orient]))
        for dimind in range(3):
            for massind in range(len(positions[orient])):
                ax[dimind, i].plot(dynamics[orient][massind][dimind], color=f"C{sind}")
                ax[dimind, i].plot(positions[orient][massind][dimind], color="k")
                sind += 1

    fig.savefig(os.path.join(root_dir, "orient_positions_1.png"))

    m_recon_tseries = np.array([
        dynamics[orient] for orient in orientations
    ])
    m_recon_masses = np.array([
        norm_masses[orient] for orient in orientations
    ])

    make_animations.make_3d_distribution(
                root_dir, 
                1, 
                m_recon_tseries, 
                m_recon_masses, 
                None, 
                None)
    

def chirp_positions(
    times, 
    upsample_times,
    m1,
    m2, 
    tc, 
    detectors=["H1", "L1", "V1"], 
    basis_order=10, 
    window="none", 
    basis_type="chebyshev", 
    root_dir=None,
    rotate_angle=0.0):

    norm_masses, positions, masses = generate_m1m2_pos(times, m1, m2, tc)

    rotation_matrix =np.array([
            [1, 0, 0],
            [0, np.cos(rotate_angle), -np.sin(rotate_angle)],
            [0, np.sin(rotate_angle), np.cos(rotate_angle)]
        ])

    positions = np.einsum("...ijk,...jm->...imk",positions,rotation_matrix)

    print("pos", np.min(positions), np.max(positions))

    basis_dynamics = fit_positions_with_polynomial(
        times, 
        positions, 
        basis_order=basis_order, 
        window=window, 
        basis_type=basis_type)

    print("bdyn", np.min(basis_dynamics), np.max(basis_dynamics))
    print(np.shape(basis_dynamics))


    # test to bring values withing training range
    print(np.max(basis_dynamics))
    max_dyn = np.max(basis_dynamics)
    norm_basis_dynamics = basis_dynamics/(1.5*max_dyn)


    print("masses",norm_masses)

    # change to mass, dim, order
    #basis_dynamics = np.transpose(coeffs, (0,2,1))

    timeseries_dynamics = compute_waveform.get_time_dynamics(
        norm_basis_dynamics, 
        times, 
        basis_type=basis_type
        )
    if root_dir is not None:
        fig, ax = plt.subplots(nrows=3)
        ax[0].plot(times, timeseries_dynamics[0,0])
        ax[1].plot(times, timeseries_dynamics[0,1])
        ax[2].plot(times, timeseries_dynamics[0,2])
        fig.savefig(os.path.join(root_dir, "chirp_positions.png"))


    strain_timeseries, energy = compute_waveform.get_waveform(
        times, 
        norm_masses, 
        norm_basis_dynamics, 
        detectors, 
        basis_type=basis_type,
        compute_energy=True)
    if root_dir is not None:
        fig, ax = plt.subplots(nrows=3)
        ax[0].plot(times, strain_timeseries[0])
        ax[1].plot(times, strain_timeseries[1])
        ax[2].plot(times, strain_timeseries[2])
        fig.savefig(os.path.join(root_dir, "chirp_strain.png"))


    return positions, norm_basis_dynamics, timeseries_dynamics, strain_timeseries, energy, max_dyn

def run_chirp_test(config, mass1=5000, mass2=5000, rotate_angle=0.0):

    
    plot_out = os.path.join(config["root_dir"], f"test_chirp_m1-{mass1}_m2-{mass2}_rotate{rotate_angle}")

    if not os.path.isdir(plot_out):
        os.makedirs(plot_out)

    basis_type = config["basis_type"]
    
    pre_model, model = load_models(config, device="cpu")
    
    times = np.linspace(0,1,config["sample_rate"])
    upsample_times = np.linspace(0,1,config["plot_sample_rate"])

    basis_order = config["basis_order"]
    #m1,m2 = 2000,500
    m1,m2 = mass1, mass2
    """
    test_different_orientations(
        times, 
        m1, 
        m2, 
        1.1, 
        basis_order, 
        config["detectors"], 
        window=config["window"],
        basis_type=config["basis_type"], 
        root_dir=plot_out)
    
    test_1and2_masses(
        times, 
        m1, 
        m2, 
        1.1, 
        basis_order, 
        config["detectors"], 
        window=config["window"],
        basis_type=config["basis_type"], 
        root_dir=plot_out)
    
    sys.exit()
    """
    dynamics, norm_basis_dynamics, all_dynamics, data, energy, max_dyn = chirp_positions(
        times, 
        upsample_times,
        m1, 
        m2, 
        1.1, 
        detectors=config["detectors"], 
        basis_order=basis_order, 
        window=config["window"], 
        root_dir=plot_out,
        basis_type=basis_type,
        rotate_angle=rotate_angle)


    basis_dynamics = norm_basis_dynamics*max_dyn
    print("basis_dynamics", norm_basis_dynamics)

    print(np.shape(basis_dynamics))

    fig, ax = plt.subplots()
    ts = np.arange(int(basis_order/2 + 1))
    ax.plot(ts, basis_dynamics[:,0,:].T)
    ax.plot(ts, basis_dynamics[:,1,:].T)
    ax.plot(ts, basis_dynamics[:,2,:].T)
    ax.plot(ts, norm_basis_dynamics[:,0,:].T, ls="--")
    ax.plot(ts, norm_basis_dynamics[:,1,:].T, ls="--")
    ax.plot(ts, norm_basis_dynamics[:,2,:].T, ls="--")
    ax.fill_between(ts, np.exp(-config["fourier_weight"]*ts)*-1, np.exp(-config["fourier_weight"]*ts)*1, alpha = 0.5)
    fig.savefig(os.path.join(plot_out, "basis_prior.png"))

    norm_data, norm_factor = data_processing.normalise_data(data, pre_model.norm_factor)    
    print("normfactor", norm_factor, np.max(data), np.max(norm_data))

    #data = data/100

    n_masses = 2
    n_dimensions = 3
   
    make_animations.make_3d_animation(plot_out, 100, all_dynamics, 0.01*np.array([m1,m2]), None, None)

    
    reconstructx = basis[basis_type]["val"](times, basis_dynamics[0][0])
    reconstructy = basis[basis_type]["val"](times, basis_dynamics[0][1])
    reconstructz = basis[basis_type]["val"](times, basis_dynamics[0][2])

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

    #print(data.shape, all_dynamics_dynamics.shape)
    source_tseries = all_dynamics
    source_masses = np.array([m1,m2])/np.sum([m1, m2])
    batch = 0

    source_tseries = compute_waveform.get_time_dynamics(
        norm_basis_dynamics, 
        upsample_times, 
        basis_type=basis_type
        )

    source_strain, source_energy = compute_waveform.get_waveform(
        upsample_times, 
        source_masses, 
        norm_basis_dynamics, 
        config["detectors"], 
        basis_type=basis_type,
        compute_energy=True)

    print("source maxmin: ", np.min(source_strain), np.max(source_strain))

    #source_strain, _ = data_processing.normalise_data(source_strain, pre_model.norm_factor)


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
            window_acceleration=config["window_acceleration"], 
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
                upsample_times, 
                config["detectors"], 
                m_recon_strain, 
                source_strain, 
                fname = os.path.join(plot_out,f"recon_strain_dist_{batch}.png"))

    plotting.plot_dimension_projection(
                m_recon_tseries[:10], 
                source_tseries, 
                fname=os.path.join(plot_out, f"dim_projection_{batch}.png"), 
                alpha=0.2)

    plotting.plot_sample_separations(
                upsample_times, 
                source_tseries, 
                m_recon_tseries, 
                fname=os.path.join(plot_out,f"separations_{batch}.png"))


    plotting.plot_mass_distributions(
                m_recon_masses,
                source_masses,
                fname=os.path.join(plot_out,f"massdistributions_{batch}.png"))

    print(np.max(m_recon_tseries), np.min(m_recon_tseries))
    make_animations.line_of_sight_animation(
                m_recon_tseries, 
                m_recon_masses, 
                source_tseries, 
                source_masses, 
                os.path.join(plot_out,f"2d_massdist_{batch}.gif"))

    print("source", np.shape(source_tseries), np.shape(source_masses))

    
    make_animations.make_3d_distribution(
                plot_out, 
                m_recon_tseries[:n_animate_samples], 
                m_recon_masses[:n_animate_samples], 
                source_tseries, 
                source_masses,
                fname = os.path.join(plot_out,f"3d_distribution_{batch}.png"))

    make_animations.heatmap_projections(
                m_recon_tseries, 
                m_recon_masses, 
                source_tseries, 
                source_masses, 
                os.path.join(plot_out,f"heatmap_projections_{batch}.gif"),
                duration=5)

    make_animations.make_distribution_projections(
                plot_out, 
                batch, 
                m_recon_tseries, 
                m_recon_masses, 
                source_tseries, 
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
    run_chirp_test(config, args.mass1, args.mass2, rotate_angle=np.pi/4)