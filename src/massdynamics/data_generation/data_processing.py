import torch
import numpy as np
import massdynamics.data_generation.compute_waveform as compute_waveform
import massdynamics.window_functions as window_functions
import scipy
import copy

def normalise_data(strain, norm_factor = None):
    """normalise the data to the maximum strain in all data

    Args:
        strain (_type_): strain array

    Returns:
        _type_: normalised strain
    """
    if norm_factor is None:
        norm_factor = np.max(strain)
    
    return np.array(strain)/norm_factor, norm_factor

def unnormalise_data(strain, norm_factor = None):
    """normalise the data to the maximum strain in all data

    Args:
        strain (_type_): strain array

    Returns:
        _type_: normalised strain
    """
    if norm_factor is None:
        norm_factor = np.max(strain)
    
    return np.array(strain)*norm_factor, norm_factor

def normalise_labels(label, label_norm_factor = None, mass_norm_factor=None,n_masses=2):
    """normalise the labels (flattened)

    Args:
        strain (_type_): strain array

    Returns:
        _type_: normalised strain
    """
    if label_norm_factor is None:
        label_norm_factor = np.max(np.abs(label[:,:-n_masses]))
        #print("nf",norm_factor, label.shape, np.max(label))

    if mass_norm_factor is None:
        mass_norm_factor = np.max(label[:,-n_masses:])

    label2 = copy.copy(label)
    label2[:,:-n_masses] /= label_norm_factor

    label2[:,-n_masses:] /= mass_norm_factor

    return np.array(label2), label_norm_factor, mass_norm_factor

def unnormalise_labels(label, label_norm_factor=None, mass_norm_factor=None, n_masses=2):
    """normalise the data to the maximum strain in all data

    Args:
        strain (_type_): strain array

    Returns:
        _type_: normalised strain
    """
    if label_norm_factor is None:
        label_norm_factor = np.max(np.abs(label[:,:-n_masses]))

    if mass_norm_factor is None:
        mass_norm_factor = np.max(label[:,-n_masses:])

    label2 = copy.copy(label)
    label2[:,:-n_masses] *= label_norm_factor

    label2[:,-n_masses:] *= mass_norm_factor
    
    return np.array(label2), label_norm_factor, mass_norm_factor

def complex_to_real(input_array):

    output_array = np.concatenate(
        [
            np.real(input_array)[..., None], 
            np.imag(input_array)[..., None]
        ], 
        axis=-1)
    
    return output_array

def real_to_complex(input_array):

    output_array = input_array[...,0] + 1j*input_array[...,1]

    return output_array

def positions_masses_to_samples(
    coeff_samples: np.array,
    mass_samples: np.array,
    basis_type: str = "chebyshev"
    ):
    """convert dynamics into samples for network

    Args:
        coeff_samples (np.array): (Nsamps, Nmasses, Ndimensions, Ncoeffs)
        mass_samples (np.array): (Nsamps, Nmasses)
        basis_order (int): _description_
        n_dimensions (int): _description_
        basis_type (str): _description_

    Returns:
        torch.Tensor: _description_
    """

    if basis_type == "fourier":
        #coeff_samples = torch.view_as_real(torch.from_numpy(coeff_samples)).flatten(start_dim=-1)
        output_coeffs = complex_to_real(coeff_samples).reshape(np.shape(coeff_samples)[0], -1)
    else:
        output_coeffs = coeff_samples.reshape(np.shape(coeff_samples)[0], -1)
    # flatten all dimensions apart from 1st which is samples
    #output_coeffs = coeff_samples.flatten(start_dim=1)
    # append masses to the flattened output coefficients 
    output_coeffs = np.concatenate([output_coeffs, mass_samples], axis=1)

    return output_coeffs


def samples_to_positions_masses(
    coeffmass_samples, 
    n_masses,
    basis_order,
    n_dimensions,
    basis_type):
    """conver outputs samples from flow into coefficients and masses

    Args:
        coeffmass_samples (torch.Tensor): (Nsamples, Ncoeffs*Nmasses*Ndimensions)
        n_masses (int): _description_
        basis_order (_type_): _description_
        n_dimensions (_type_): _description_
        basis_type (_type_): _description_

    Returns:
        _type_: masses (Nsamples, Nmasses), coeffs (Nsamples, Nmasses, Ndimensions, Ncoeffs)
    """
    masses = coeffmass_samples[:, -n_masses:]
    if basis_type == "fourier":
        sshape = np.shape(coeffmass_samples[:, :-n_masses])

        coeffs = coeffmass_samples[:, :-n_masses].reshape(
            sshape[0], 
            n_masses, 
            n_dimensions, 
            int(basis_order/2 + 1), 
            2)
        #coeffs = torch.view_as_complex(coeffs).numpy()
        coeffs = real_to_complex(coeffs)
        # plus 1 on basis order as increased coeffs to return same ts samples
        # also divide half as half basis size when complex number
        #coeffs = np.transpose(coeffs, (0,1,3,2))
        #coeffs = coeffs.reshape(sshape[0], n_masses, int(0.5*basis_order+1), n_dimensions)
    else:
        coeffs = coeffmass_samples[:,:-n_masses].reshape(-1,n_masses, n_dimensions, basis_order)

    return masses, coeffs


def get_strain_from_samples(
    times, 
    masses, 
    coeffs, 
    detectors=["H1"],
    window_acceleration=False, 
    window="none", 
    basis_type="chebyshev",
    basis_order=16,
    sky_position=(np.pi, np.pi/2)):
    """_summary_

    Args:
        times (_type_): _description_
        masses (_type_): _description_
        coeffs (_type_): (n_masses, n_coeffs, n_dimensions)
        detectors (list, optional): _description_. Defaults to ["H1"].
        window_acceleration (bool, optional): _description_. Defaults to False.
        window (bool, optional): _description_. Defaults to False.
        basis_type (str, optional): _description_. Defaults to "chebyshev".

    Returns:
        _type_: _description_
    """
    # if there is a window and I and not predicting the windowed coefficients
   
    n_masses, n_dimensions, n_coeffs = np.shape(coeffs)
    """
    if window_acceleration not in [False, None, "none"]:
        n_coeffs = []
        # for each mass perform the window on the xyz positions (acceleration)
        for mass in range(n_masses):
            temp_recon, win_coeffs = window_functions.perform_window(times, coeffs[mass].T, window_acceleration, basis_type=basis_type, order=basis_order)
            n_coeffs.append(temp_recon.T)
        
        # update the coefficients with the windowed version
        coeffs = np.array(n_coeffs)
    """

    strain, energy = compute_waveform.get_waveform(
        times, 
        masses, 
        coeffs, 
        detectors, 
        basis_type=basis_type,
        compute_energy=True,
        sky_position=sky_position
    )


    return strain, energy, coeffs

def get_window_strain(strain, window_type="hann", alpha=0.1):
    """Window the strain

    Args:
        strain (_type_): _description_
        window_type (str, optional): _description_. Defaults to "hann".
        alpha (float, optional): _description_. Defaults to 0.1.

    Returns:
        _type_: _description_
    """
    if window_type == "hann":
        window = np.hanning(np.shape(strain)[-1])
        win_strain = strain * window
    elif window_type == "tukey":
        window = scipy.signal.windows.tukey(np.shape(strain)[-1], alpha=0.1)
        win_strain = strain * window
    else:
        win_strain = strain

    return win_strain

def subtract_center_of_mass(positions, masses):
    """
    Subtract the center of mass motion from a time series of N masses.

    Parameters:
    - positions: Numpy array of shape (n_masses, n_dimensions, time) representing the positions of two masses over time.
    - masses: Numpy array of shape (n_masses,) representing the masses of the two objects.

    Returns:
    - relative_positions: Numpy array of shape (n_masses, n_dimensions, time) representing the relative positions after subtracting
      the center of mass motion.
    """
    # Calculate the center of mass position at each time step
    center_of_mass_positions = np.average(positions, axis=0, weights=masses)

    # Subtract the center of mass position from the positions of the individual masses
    relative_positions = positions - center_of_mass_positions[np.newaxis, :, :]

    return relative_positions

def compute_angular_momentum(m1, m2, positions, velocities):
    """
    Compute the total angular momentum of the system.
    
    Parameters:
    m1, m2: masses of the two particles
    positions: array of shape (N, 2, 3) representing the positions of the particles over time
    velocities: array of shape (N, 2, 3) representing the velocities of the particles over time
    
    Returns:
    L: array of shape (N, 3) representing the angular momentum at each time step
    """
    r1, r2 = positions[:, 0, :], positions[:, 1, :]
    v1, v2 = velocities[:, 0, :], velocities[:, 1, :]
    
    # Calculate the angular momentum for each particle and sum them up
    L = m1 * np.cross(r1, v1) + m2 * np.cross(r2, v2)
    return L

def conserve_angular_momentum(m1, m2, positions, velocities):
    """shift the positions such that they conserve angular momentum

    Args:
        m1 (_type_): _description_
        m2 (_type_): _description_
        positions (_type_): _description_
        velocities (_type_): _description_
    """

    


def cartesian_to_spherical(positions):
    """Convert cartesian to spherical coords

    Args:
        positions (_type_): input shape (Ndata, Ndimensions, Nsamples)

    Returns:
        _type_: (Ndata, Ndimensions, Nsamples)
    """
    r = np.sqrt(np.einsum("ijk,ijk->ik", positions, positions))
    theta = np.unwrap(np.arctan2(np.einsum("ijk,ijk->ik", positions[:,:2,:], positions[:,:2,:]), positions[:,2,:]), axis=-1)
    phi = np.unwrap(np.arctan2(positions[:,1,:], positions[:,0,:]), axis=-1)

    positions_spherical = np.concatenate([r[:,np.newaxis,:], phi[:,np.newaxis,:], theta[:,np.newaxis,:]], axis=1)

    return positions_spherical

def spherical_to_cartesian(positions):
    """Convert cartesian to spherical coords

    Args:
        positions (_type_): input shape (Ndata, Ndimensions, Nsamples)

    Returns:
        _type_: (Ndata, Ndimensions, Nsamples)
    """
    x = positions[:,0,:] * np.sin(positions[:,2,:]) * np.cos(positions[:,1,:])
    y = positions[:,0,:] * np.sin(positions[:,2,:]) * np.sin(positions[:,1,:])
    z = positions[:,0,:] * np.cos(positions[:,2,:])

    positions_spherical = np.concatenate([x[:,np.newaxis,:], y[:,np.newaxis,:], z[:,np.newaxis,:]], axis=1)

    return positions_spherical


def preprocess_data(
    pre_model, 
    basis_dynamics,
    masses, 
    strain, 
    window_strain=None, 
    spherical_coords=None, 
    initial_run=False,
    n_masses=2,
    device="cpu",
    basis_type="fourier",
    n_dimensions=3):

    if spherical_coords:
        print("Spherical not implemented yet")
        #time_dynamics = cartesian_to_spherical(time_dynamics)



    if window_strain is not None:
        strain = get_window_strain(strain, window_type=window_strain)
    
    # get only the required dimensions for training
    #print("b1", np.shape(basis_dynamics))
    basis_dynamics = basis_dynamics[...,:n_dimensions,:]
    #print("b2", np.shape(basis_dynamics))
    
    labels = positions_masses_to_samples(
        basis_dynamics,
        masses,
        basis_type = basis_type
        )

    if initial_run:
        strain, norm_factor = normalise_data(
            strain, 
            None)
        pre_model.norm_factor = norm_factor
        labels, label_norm_factor, mass_norm_factor = normalise_labels(
            labels, 
            None, 
            None,
            n_masses=n_masses)
        pre_model.label_norm_factor = label_norm_factor
        pre_model.mass_norm_factor = mass_norm_factor
    else:
        strain, norm_factor = normalise_data(
            strain, 
            pre_model.norm_factor)
        labels, label_norm_factor, mass_norm_factor = normalise_labels(
            labels, 
            label_norm_factor=pre_model.label_norm_factor, 
            mass_norm_factor=pre_model.mass_norm_factor, 
            n_masses=n_masses)


    return pre_model, labels, strain

def unpreprocess_data(
    pre_model, 
    labels, 
    strain,
    window_strain=None, 
    spherical_coords=None, 
    initial_run=False,
    n_masses=2,
    n_dimensions=3,
    basis_type="fourier",
    basis_order=16,
    device="cpu"):


    if spherical_coords:
        print("spherical not implemented yet")
        #basis_dynamics = spherical_to_cartesian(basis_dynamics)
    
    strain, norm_factor = unnormalise_data(
        strain, 
        pre_model.norm_factor)

    labels2, label_norm_factor, mass_norm_factor = unnormalise_labels(
        labels, 
        label_norm_factor=pre_model.label_norm_factor, 
        mass_norm_factor=pre_model.mass_norm_factor, 
        n_masses=n_masses)

    masses, basis_dynamics = samples_to_positions_masses(
            labels2,
            n_masses,
            basis_order,
            n_dimensions,
            basis_type
        )
    
    if n_dimensions != 3:
        bd_shape = list(np.shape(basis_dynamics))
        bd_shape[-2] = 3 - n_dimensions
        basis_dynamics = np.concatenate([basis_dynamics, np.zeros(bd_shape)], axis=-2)

    return pre_model, masses, basis_dynamics, strain
