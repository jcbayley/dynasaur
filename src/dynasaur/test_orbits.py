import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import os
#from dynasaur.data_generation import compute_hTT_coeffs, compute_strain_from_coeffs, perform_window, polynomial_dict, compute_energy_loss
import dynasaur.data_generation
#import make_animations as animations
import dynasaur.plotting
#from dynasaur.train_model import get_dynamics, get_strain_from_samples
from dynasaur.create_model import create_models, load_models
from dynasaur.data_generation import (
    data_generation,
    compute_waveform,
    data_processing
)

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

    norm_masses = np.array([m1, m2])/np.sum([m1,m2])

    return norm_masses, positions

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
    #chebyshev_order = 30

    ph_offset = 0
    x = np.sin(2*np.pi*times*1 + ph_offset)
    y = np.sin(2*np.pi*times*1 + ph_offset)
    z = np.zeros(len(times))


    if orientation == "xyz":
        m1pos = np.vstack([x,y,z])
        #m2pos = np.vstack([x,y,z])
    elif orientation == "x-yz":
        m1pos = np.vstack([x,-y,z])
        #m2pos = np.vstack([x,y,z])
    elif orientation == "-x-yz":
        m1pos = np.vstack([-x,-y,z])
        #m2pos = np.vstack([x,y,z])
    elif orientation == "xzy":
        m1pos = np.vstack([x,z,y])
        #m2pos = np.vstack([x,y,z])
    elif orientation == "offz":
        m1pos = np.vstack([x,y,z+0.2])
        #m2pos = np.vstack([x,y,z])
    elif orientation == "x00":
        m1pos = np.vstack([x,z,z])
        #m2pos = np.vstack([x,y,z])
    elif orientation == "0y0":
        m1pos = np.vstack([z,x,z])
        #m2pos = np.vstack([x,y,z])
    elif orientation == "00z":
        m1pos = np.vstack([z,z,x])
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

    positions = positions/np.max(positions)

    norm_masses = np.array([m1,])/np.sum([m1,])

    return norm_masses, positions

def fit_positions_with_polynomial(times, positions, chebyshev_order=8, window_acceleration="none", poly_type="chebyshev"):
    """fit the 3d positions with a polynomial

    Args:
        times (_type_): _description_
        positions (_type_): _description_
        chebyshev_order (int, optional): _description_. Defaults to 8.
        window (bool, optional): _description_. Defaults to False.
        poly_type (str, optional): _description_. Defaults to "chebyshev".

    Returns:
        _type_: _description_
    """
  
    n_masses, n_dimensions, n_cheby = np.shape(positions)

    if window_acceleration not in [False, None,"none"]:
        cheb_dynamics = np.zeros((n_masses, n_dimensions, chebyshev_order))
    else:
        cheb_dynamics = []

    for mind in range(n_masses):
        temp_dyn = np.zeros((chebyshev_order, n_dimensions))
        for dimind in range(3):
            temp_dyn[:,dimind] = polynomial_dict[poly_type]["fit"](
                times, 
                positions[mind, dimind], 
                chebyshev_order-1)

        if window_acceleration not in [False, None,"none"]:
            temp_dyn2, win_coeffs = perform_window(
                times, 
                temp_dyn, 
                window_acceleration, 
                order=chebyshev_order, 
                poly_type=poly_type)
            print(np.shape(temp_dyn), np.shape(temp_dyn2))
            cheb_dynamics.append(temp_dyn2.T)
        else:
            cheb_dynamics[mind] = temp_dyn.T


    cheb_dynamics = np.array(cheb_dynamics)

    return cheb_dynamics

def get_ts_dynamics(times, cheb_dynamics, poly_type="chebyshev"):

    n_masses, n_dimensions, n_coeffs = np.shape(cheb_dynamics)
    timeseries_dynamics = np.zeros((n_masses,n_dimensions,len(times)))
    for i in range(n_masses):
        for j in range(n_dimensions):
            timeseries_dynamics[i,j] = polynomial_dict[poly_type]["val"](times, cheb_dynamics[i,j])
        
    return timeseries_dynamics

def get_waveform(times, norm_masses, cheb_dynamics, detectors, poly_type="chebyshev"):

    strain_coeffs = compute_hTT_coeffs(norm_masses, cheb_dynamics, poly_type=poly_type)

    energy = compute_energy_loss(times, norm_masses, cheb_dynamics, poly_type=poly_type)

    strain_timeseries = np.zeros((len(detectors), len(times)))
    for dind, detector in enumerate(detectors):
        strain = compute_strain_from_coeffs(times, strain_coeffs, detector=detector, poly_type=poly_type)
        #strain = compute_strain(temp_strain_timeseries, detector, poly_type=poly_type)
        strain_timeseries[dind] = strain
    
    return strain_timeseries, energy

def test_different_orientations(times, m1, m2, tc, chebyshev_order, detectors, window="none", poly_type="chebyshev", root_dir="./"):

    orientations = ["xyz", "x-yz", "-x-yz", "xzy", "offz"]
    #orientations = ["x00", "0y0", "00z",]
    positions = {}
    strain_timeseries = {}
    energy = {}
    cheb_dynamics = {}
    norm_masses = {}
    dynamics = {}
    for orient in orientations:
           
        norm_masses[orient], positions[orient] = generate_m1m2_pos_1d(
            times, 
            m1, 
            m2, 
            tc, 
            orientation=orient)

        #if orient == "xz":
        #    norm_masses["xz"] = 1.5*norm_masses["xz"]
        cheb_dynamics[orient] = fit_positions_with_polynomial(
            times, 
            positions[orient], 
            chebyshev_order=chebyshev_order, 
            window=window, 
            poly_type=poly_type)
        
        dynamics[orient] = get_ts_dynamics(
            times, 
            cheb_dynamics[orient], 
            poly_type=poly_type)
        
        print(np.shape(cheb_dynamics[orient]))

        strain_timeseries[orient], energy[orient] = get_waveform(
            times, 
            norm_masses[orient], 
            cheb_dynamics[orient], 
            detectors, 
            poly_type=poly_type)

    fig, ax = plt.subplots(nrows = len(detectors) + 1)
    lss = ["-", "--", "-.", ":", "--"]
    for oind, orient in enumerate(orientations):
        for detind in range(len(detectors)):
            ax[detind].plot(strain_timeseries[orient][detind], label=f"or: {orient}", ls=lss[oind])

            ax[detind].legend(loc="upper left")
    fig.savefig(os.path.join(root_dir, "orient_strain_2.png"))

    fig, ax = plt.subplots(nrows = 3, ncols=len(orientations))
    sind = 0
    for i,orient in enumerate(orientations):
        print(np.shape(dynamics[orient]))
        print(np.shape(positions[orient]))
        for dimind in range(3):
            ax[dimind, i].plot(dynamics[orient][0][dimind], color=f"C{sind}", label=f"or: {orient}")
            ax[dimind, i].plot(positions[orient][0][dimind], color="k")
            ax[dimind, i].legend(loc="upper left")
            sind += 1

    fig.savefig(os.path.join(root_dir, "orient_positions_2.png"))

    m_recon_tseries = np.array([
        dynamics[orient] for orient in orientations
    ])
    m_recon_masses = np.array([
        norm_masses[orient] for orient in orientations
    ])


    animations.make_3d_distribution(
                root_dir, 
                0, 
                m_recon_tseries, 
                m_recon_masses, 
                None, 
                None)
    
def test_1and2_masses(times, m1, m2, tc, chebyshev_order, detectors, window="none", poly_type="chebyshev", root_dir="./"):

    orientations = ["twomass", "onemass", "othermass"]
    positions = {}
    strain_timeseries = {}
    energy = {}
    cheb_dynamics = {}
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
        cheb_dynamics[orient] = fit_positions_with_polynomial(
            times, 
            positions[orient], 
            chebyshev_order=chebyshev_order, 
            window=window, 
            poly_type=poly_type)
        
        dynamics[orient] = get_ts_dynamics(
            times, 
            cheb_dynamics[orient], 
            poly_type=poly_type)
        
        print(np.shape(cheb_dynamics[orient]))

        strain_timeseries[orient], energy[orient] = get_waveform(
            times, 
            norm_masses[orient], 
            cheb_dynamics[orient], 
            detectors, 
            poly_type=poly_type)

    fig, ax = plt.subplots(nrows = len(detectors) + 1)
    
    lss = ["-", "--", "-.", ":"]
    for oind,orient in enumerate(orientations):
        for detind in range(len(detectors)):
            ax[detind].plot(strain_timeseries[orient][detind], ls=lss[oind], label=f"or: {orient}")
            ax[detind].legend(loc="upper left")
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
                ax[dimind, i].plot(dynamics[orient][massind][dimind], color=f"C{sind}", label=f"or: {orient}")
                ax[dimind, i].plot(positions[orient][massind][dimind], color="k")
                ax[dimind, i].legend(loc="upper left")
                sind += 1

    fig.savefig(os.path.join(root_dir, "orient_positions_1.png"))

    m_recon_tseries = np.array([
        dynamics[orient] for orient in orientations
    ])
    m_recon_masses = np.array([
        norm_masses[orient] for orient in orientations
    ])

    animations.make_3d_distribution(
                root_dir, 
                1, 
                m_recon_tseries, 
                m_recon_masses, 
                None, 
                None)
    

def chirp_positions(times, m1, m2, tc, detectors=["H1", "L1", "V1"], chebyshev_order=10, window="none", poly_type="chebyshev", root_dir="./"):

    norm_masses, positions = generate_m1m2_pos(times, m1, m2, tc)

    cheb_dynamics = fit_positions_with_polynomial(
        times, 
        positions, 
        chebyshev_order=chebyshev_order, 
        window=window, 
        poly_type=poly_type)


    print("masses",norm_masses)

    # change to mass, dim, order
    #cheb_dynamics = np.transpose(coeffs, (0,2,1))

    timeseries_dynamics = get_ts_dynamics(times, cheb_dynamics, poly_type=poly_type)

    fig, ax = plt.subplots(nrows=3)
    ax[0].plot(times, timeseries_dynamics[0,0])
    ax[1].plot(times, timeseries_dynamics[0,1])
    ax[2].plot(times, timeseries_dynamics[0,2])
    fig.savefig(os.path.join(root_dir, "chirp_positions.png"))


    strain_timeseries, energy = get_waveform(times, norm_masses, cheb_dynamics, detectors, poly_type="chebyshev")

    fig, ax = plt.subplots(nrows=3)
    ax[0].plot(times, strain_timeseries[0])
    ax[1].plot(times, strain_timeseries[1])
    ax[2].plot(times, strain_timeseries[2])
    fig.savefig(os.path.join(root_dir, "chirp_strain.png"))


    return positions, cheb_dynamics, timeseries_dynamics, strain_timeseries, energy

def run_chirp_test(config, mass1=5000, mass2=5000):

    
    plot_out = os.path.join(config["root_dir"], f"orbit_test")

    if not os.path.isdir(plot_out):
        os.makedirs(plot_out)

    poly_type = "chebyshev"
    
    #pre_model, model = load_models(config, device="cpu")
    
    times = np.linspace(-1,1,config["sample_rate"])

    chebyshev_order = config["chebyshev_order"]
    #m1,m2 = 2000,500
    m1,m2 = mass1, mass2
    test_different_orientations(
        times, 
        m1, 
        m2, 
        1.1, 
        chebyshev_order, 
        config["detectors"], 
        window=config["window"],
        poly_type=config["poly_type"], 
        root_dir=plot_out)
    
    test_1and2_masses(
        times, 
        m1, 
        m2, 
        1.1, 
        chebyshev_order, 
        config["detectors"], 
        window=config["window"],
        poly_type=config["poly_type"], 
        root_dir=plot_out)
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, required=False, default="none")
    parser.add_argument("-m1", "--mass2", type=float, required=False, default=5000)
    parser.add_argument("-m2", "--mass1", type=float, required=False, default=5000)
    args = parser.parse_args()


    with open(os.path.abspath(args.config), "r") as f:
        config = json.load(f)


    run_chirp_test(config, args.mass1, args.mass2)