import torch
import numpy as np
import dynasaur.data_generation.compute_waveform as compute_waveform
import dynasaur.window_functions as window_functions
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
        if type(strain) == np.ndarray:
            norm_factor = np.max(strain)
        else:
            norm_factor = torch.max(strain).to("cpu").numpy()

    if type(norm_factor) == torch.Tensor:
        norm_factor = norm_factor.to("cpu").numpy()

    return strain/norm_factor, norm_factor

def unnormalise_data(strain, norm_factor = None):
    """normalise the data to the maximum strain in all data

    Args:
        strain (_type_): strain array

    Returns:
        _type_: normalised strain
    """
    if norm_factor is None:
        if type(strain) == np.ndarray:
            norm_factor = np.max(strain)
        else:
            norm_factor = torch.max(strain).to("cpu").numpy()

    if type(norm_factor) == torch.Tensor:
        norm_factor = norm_factor.to("cpu").numpy()
    
    return strain*norm_factor, norm_factor

def normalise_labels(label, label_norm_factor = None, mass_norm_factor=None,n_masses=2):
    """normalise the labels (flattened)

    Args:
        strain (_type_): strain array

    Returns:
        _type_: normalised strain
    """
    if label_norm_factor is None:
        if type(label) == np.ndarray:
            label_norm_factor = np.max(np.abs(label[:,:-n_masses]))
        else:
            label_norm_factor = torch.max(torch.abs(label[:,:-n_masses]))
        #print("nf",norm_factor, label.shape, np.max(label))

    if mass_norm_factor is None:
        if type(label) == np.ndarray:
            mass_norm_factor = np.max(label[:,-n_masses:])
        else:
            mass_norm_factor = torch.max(label[:,-n_masses:])

    label2 = copy.copy(label)
    label2[:,:-n_masses] /= label_norm_factor

    label2[:,-n_masses:] /= mass_norm_factor

    return label2, label_norm_factor, mass_norm_factor

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
    
    return label2, label_norm_factor, mass_norm_factor

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
    basis_type: str = "chebyshev",
    velocities = None,
    accelerations = None,
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
        if velocities is not None:
            output_vels = complex_to_real(velocities).reshape(np.shape(coeff_samples)[0], -1)
        if accelerations is not None:
            output_accs = complex_to_real(accelerations).reshape(np.shape(coeff_samples)[0], -1)
    else:
        output_coeffs = coeff_samples.reshape(np.shape(coeff_samples)[0], -1)
        if velocities is not None:
            output_vels = velocities.reshape(np.shape(coeff_samples)[0], -1)
        if accelerations is not None:
            output_accs = accelerations.reshape(np.shape(coeff_samples)[0], -1)
    # flatten all dimensions apart from 1st which is samples
    #output_coeffs = coeff_samples.flatten(start_dim=1)
    # append masses to the flattened output coefficients 
    if velocities is not None:
        output_coeffs = np.concatenate([output_coeffs, output_vels], axis=1)
    if accelerations is not None:
        output_coeffs = np.concatenate([output_coeffs, output_accs], axis=1)

    output_coeffs = np.concatenate([output_coeffs, mass_samples], axis=1)
    return output_coeffs


def samples_to_positions_masses(
    coeffmass_samples, 
    n_masses,
    basis_order,
    n_dimensions,
    basis_type,
    includes_velocities=False):
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

        if includes_velocities:
            coeffs = coeffmass_samples[:, :-n_masses].reshape(
                sshape[0], 
                n_masses, 
                n_dimensions, 
                2,
                int(basis_order/2 + 1), 
                2)
            velocities = coeffs[:,:,:,1]
            coeffs = coeffs[:,:,:,0]
            velocities = real_to_complex(velocities)
        else:
            coeffs = coeffmass_samples[:, :-n_masses].reshape(
                sshape[0], 
                n_masses, 
                n_dimensions, 
                int(basis_order/2 + 1), 
                2)
            velocities = None
        #coeffs = torch.view_as_complex(coeffs).numpy()
        coeffs = real_to_complex(coeffs)
        # plus 1 on basis order as increased coeffs to return same ts samples
        # also divide half as half basis size when complex number
        #coeffs = np.transpose(coeffs, (0,1,3,2))
        #coeffs = coeffs.reshape(sshape[0], n_masses, int(0.5*basis_order+1), n_dimensions)
    else:
        if includes_velocities:
            coeffs = coeffmass_samples[:,:-n_masses].reshape(-1,n_masses, n_dimensions, 2, basis_order)
            velocities = coeffs[:,:,:,1]
            coeffs = coeffs[:,:,:,0]
        else:
            coeffs = coeffmass_samples[:,:-n_masses].reshape(-1,n_masses, n_dimensions, basis_order)
            velocities = None
    return masses, coeffs, velocities


def get_strain_from_samples(
    times, 
    masses, 
    coeffs, 
    detectors=["H1"],
    window_acceleration=False, 
    window_strain="none", 
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

def reshape_to_time_batch(tensor):
    """
    Reshape a tensor from shape (batch, n_mass, n_dim, n_time) to (batch * n_time, n_mass, n_dim).
    Parameters:
    tensor (torch.Tensor): Input tensor of shape (batch, n_mass, n_dim, n_time)
    Returns:
    torch.Tensor: Reshaped tensor of shape (batch * n_time, n_mass, n_dim)
    """
    batch, n_mass, n_dim, n_time = tensor.shape
    return tensor.permute(0, 3, 1, 2).reshape(batch * n_time, n_mass, n_dim)

def reshape_to_original(tensor, batch, n_time):
    """
    Reshape a tensor from shape (batch * n_time, n_mass, n_dim) to (batch, n_mass, n_dim, n_time).
    Parameters:
    tensor (torch.Tensor): Input tensor of shape (batch * n_time, n_mass, n_dim)
    batch (int): Original batch size
    n_time (int): Original n_time size
    Returns:
    torch.Tensor: Reshaped tensor of shape (batch, n_mass, n_dim, n_time)
    """
    n_mass, n_dim = tensor.shape[1], tensor.shape[2]
    return tensor.reshape(batch, n_time, n_mass, n_dim).permute(0, 2, 3, 1)


def split_time_data(basis_dynamics, strain, masses, n_previous_positions=None, basis_velocities=None, basis_accelerations=None):

    basis_dynamics = torch.from_numpy(basis_dynamics) 
    strain = torch.from_numpy(strain)
    masses = torch.from_numpy(masses)
    if basis_velocities is not None:
        basis_velocities = torch.from_numpy(basis_velocities)
    if basis_accelerations is not None:
        basis_accelerations = torch.from_numpy(basis_accelerations)

    batch_size, n_m, n_d, n_t = basis_dynamics.shape
    # shape [batch_size, n_masses, n_dimensions, n_timesteps]
    # reshape to [batch_size*n_timesteps, n_masses, n_dimensions]

    split_dynamics = reshape_to_time_batch(basis_dynamics)
    if basis_velocities is not None:
        split_velocities = reshape_to_time_batch(basis_velocities)
    else:
        split_velocities = None

    if basis_accelerations is not None:
        split_accelerations = reshape_to_time_batch(basis_accelerations)
    else:
        split_accelerations = None
    # make strain shape (batchsize*n_times, n_detectors, n_timesteps)
    #strain = strain.repeat((n_t, 1, 1))#
    strain = strain.repeat_interleave(n_t, dim=0)
    #masses = masses.repeat((n_t, 1))
    masses = masses.repeat_interleave(n_t, dim=0)
    batch_times = torch.linspace(0,1,n_t).repeat(batch_size)

    if n_previous_positions not in ["none", None, False]:
        # repeat basis dynamics n_time times
        # define indices as the index minus 2 data points
        indices = torch.stack([torch.arange(i-n_previous_positions, i) for i in range(n_t)])
        # for first point just keep selecting the same point
        #indices[indices<0] = 0
        # select indices to use and move them to the second dimension equivalent to above dynamics, then flatten as before
        previous_positions = basis_dynamics[:,:,:,indices]
        # now has shape (batch_size, n_masses, n_dimensions, n_timesteps, n_prev_points)
        previous_positions[:,:,:,indices<0] = 0
        # add noise to the previous positions as on evaluation we do not have the absolute truth.
        previous_positions += torch.randn(previous_positions.size())*0.1
        previous_positions = previous_positions.permute(0,3,1,2,4).reshape(batch_size*n_t, n_m, n_d, n_previous_positions)
    else:
        previous_positions = torch.zeros((np.shape(split_dynamics)[0], 1))

    return batch_times.numpy(), strain.numpy(), masses.numpy(), previous_positions.numpy(), split_dynamics.numpy(), split_velocities.numpy() if split_velocities is not None else None, split_accelerations.numpy() if split_accelerations is not None else None

def unsplit_time_data(labels, strain, n_masses, n_dimensions, basis_order, return_accelerations=False, return_velocities=False):
    """
    Splits the time data into batches and extracts relevant information.
    Args:
        labels (ndarray): Array of labels.
        strain (ndarray): Array of strain data.
        n_masses (int): Number of masses.
        n_dimensions (int): Number of dimensions.
        basis_order (int): Basis order.
        return_accelerations (bool, optional): Whether to return accelerations. Defaults to False.
        return_velocities (bool, optional): Whether to return velocities. Defaults to False.
    Returns:
        tuple: A tuple containing the following arrays:
            - strain (ndarray): Strain data.
            - masses (ndarray): Masses data.
            - basis_dynamics (ndarray): Basis dynamics data.
            - basis_velocities (ndarray, optional): Basis velocities data. None if return_velocities is False.
            - basis_accelerations (ndarray, optional): Basis accelerations data. None if return_accelerations is False.
    """

    # split each timestep into a different batch
    n_batchtimes, n_detectors, n_times = np.shape(strain)
    n_batch = n_batchtimes//n_times

    strain = torch.from_numpy(strain)
    labels = torch.from_numpy(labels)
    
    #print(strain.size())
    strain = strain.view((n_batch, n_times, n_detectors, n_times))[:, 0, :, :]
    n_samp_batch = labels.shape[0]//n_times
    masses = labels[:,-n_masses:].view((n_samp_batch, n_times, n_masses))[:, 0, :]
    strain = strain
    masses = masses

    # if velocities included, split them from positionsal coeffs
    if return_velocities and return_accelerations:
        basis_m_dynamics = labels[:,:-n_masses].reshape(-1, 3, n_masses, n_dimensions)
        basis_dynamics = basis_m_dynamics[:,0,:,:]
        basis_velocities = basis_m_dynamics[:,1,:,:]
        basis_accelerations = basis_m_dynamics[:,2,:,:]
    elif return_accelerations and not return_velocities:
        basis_m_dynamics = labels[:,:-n_masses].reshape(-1, 2, n_masses, n_dimensions)
        basis_dynamics = basis_m_dynamics[:,0,:,:]
        basis_accelerations = basis_m_dynamics[:,1,:,:]
    elif return_velocities and not return_accelerations:
        basis_m_dynamics = labels[:,:-n_masses].reshape(-1, 2, n_masses, n_dimensions)
        basis_dynamics = basis_m_dynamics[:,0,:,:]
        basis_velocities = basis_m_dynamics[:,1,:,:]
    else:
        basis_dynamics = labels[:,:-n_masses].reshape(-1, n_masses, n_dimensions)
    
    n_samp_batch = basis_dynamics.shape[0]//basis_order
    basis_dynamics = reshape_to_original(basis_dynamics, n_samp_batch, basis_order)
    if return_velocities:
        basis_velocities = reshape_to_original(basis_velocities, n_samp_batch, basis_order)
    else:
        basis_velocities = None

    if return_accelerations:
        basis_accelerations = reshape_to_original(basis_accelerations, n_samp_batch, basis_order)
    else:
        basis_accelerations = None
    #print(np.shape(basis_dynamics))

    return strain.numpy(), masses.numpy(), basis_dynamics.numpy(), basis_velocities.numpy() if basis_velocities is not None else None, basis_accelerations.numpy() if basis_accelerations is not None else None

def preprocess_data(
    pre_model, 
    basis_dynamics,
    masses, 
    strain, 
    basis_velocities=None,
    basis_accelerations=None,
    window_strain=None, 
    spherical_coords=None, 
    initial_run=False,
    n_masses=2,
    device="cpu",
    basis_type="fourier",
    n_dimensions=3,
    split_data=False,
    n_previous_positions="none"):

    if spherical_coords:
        print("Spherical not implemented yet")
        #time_dynamics = cartesian_to_spherical(time_dynamics)


    if window_strain not in ["none", None, False]:
        strain = get_window_strain(strain, window_type=window_strain)
    """
    strain = torch.from_numpy(strain).to(torch.float32).to(device)
    masses = torch.from_numpy(masses).to(torch.float32).to(device)
    # get only the required dimensions for training
    #print("b1", np.shape(basis_dynamics))
    basis_dynamics = torch.from_numpy(basis_dynamics[...,:n_dimensions,:]).to(torch.float32).to(device)
    if basis_velocities is not None:
        basis_velocities = torch.from_numpy(basis_velocities[...,:n_dimensions,:]).to(torch.float32).to(device)
    if basis_accelerations is not None:
        basis_accelerations = torch.from_numpy(basis_accelerations[...,:n_dimensions,:]).to(torch.float32).to(device)
    #print("b2", np.shape(basis_dynamics))
    """
    basis_dynamics = basis_dynamics[...,:n_dimensions,:]
    if basis_velocities is not None:
        basis_velocities = basis_velocities[...,:n_dimensions,:]
    if basis_accelerations is not None:
        basis_accelerations = basis_accelerations[...,:n_dimensions,:]

    if split_data:
        batch_times, strain, masses, previous_positions, split_dynamics, split_velocities, split_accelerations = split_time_data(
            basis_dynamics, 
            strain, 
            masses, 
            n_previous_positions, 
            basis_velocities, 
            basis_accelerations)

    else:
        batch_times = None
        previous_positions = None
        split_dynamics = basis_dynamics
        split_velocities = basis_velocities
        split_accelerations = basis_accelerations
    
    #print("basis vel shapes:", np.shape(split_dynamics), np.shape(split_velocities))
    labels = positions_masses_to_samples(
        split_dynamics,
        masses,
        basis_type = basis_type,
        velocities = split_velocities,
        accelerations = split_accelerations
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


    return pre_model, labels, strain, batch_times, previous_positions

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
    device="cpu",
    split_data=False,
    return_velocities=False,
    return_accelerations=False):


    if spherical_coords:
        print("spherical not implemented yet")
        #basis_dynamics = spherical_to_cartesian(basis_dynamics)
    
    if strain is not None:
        # remove normalisation from strain
        strain, norm_factor = unnormalise_data(
            strain, 
            pre_model.norm_factor)

    if labels is not None:
        # remove normalisation from the labels (different normalisation for positionsal and masses)
        labels2, label_norm_factor, mass_norm_factor = unnormalise_labels(
            labels, 
            label_norm_factor=pre_model.label_norm_factor, 
            mass_norm_factor=pre_model.mass_norm_factor, 
            n_masses=n_masses)
        #labels2 = torch.Tensor(labels2)

    else:
        masses, basis_dynamics = None, None
    
    if split_data:
        strain, masses, basis_dynamics, basis_velocities, basis_accelerations = unsplit_time_data(labels2, strain, n_masses, n_dimensions, basis_order, return_accelerations, return_velocities)
    else:
        # convert the samples into separate positions and masses
        masses, basis_dynamics = samples_to_positions_masses(
                labels2,
                n_masses,
                basis_order,
                n_dimensions,
                basis_type
            )
        basis_velocities = None
    
    # if the number of dimensions is less than 3 add extra dimensions to make it 3 (filled with zeros)
    if n_dimensions != 3:
        bd_shape = list(np.shape(basis_dynamics))
        bd_shape[-2] = 3 - n_dimensions
        basis_dynamics = np.concatenate([basis_dynamics, np.zeros(bd_shape)], axis=-2)

    return pre_model, masses, basis_dynamics, strain, basis_velocities, basis_accelerations


