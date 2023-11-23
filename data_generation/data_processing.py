import torch
import numpy as np
import compute_waveform
import ..window_functions

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
        coeff_samples = torch.view_as_real(torch.from_numpy(coeff_samples)).flatten(start_dim=-1)
    else:
        coeff_samples = torch.from_numpy(coeff_samples)
    # flatten all dimensions apart from 1st which is samples
    output_coeffs = coeff_samples.flatten(start_dim=1)
    # append masses to the flattened output coefficients 
    output_coeffs = torch.cat([output_coeffs, torch.from_numpy(mass_samples)], dim=1)

    return output_coeffs.numpy()

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
    masses = coeffmass_samples[:, -n_masses:].numpy()
    if basis_type == "fourier":
        sshape = np.shape(coeffmass_samples[:, :-n_masses])
        coeffs = coeffmass_samples[:, :-n_masses].reshape(sshape[0], n_masses, n_dimensions, int(basis_order/2 + 1), 2)
        coeffs = torch.view_as_complex(coeffs).numpy()
        # plus 1 on basis order as increased coeffs to return same ts samples
        # also divide half as half basis size when complex number
        #coeffs = np.transpose(coeffs, (0,1,3,2))
        #coeffs = coeffs.reshape(sshape[0], n_masses, int(0.5*basis_order+1), n_dimensions)
    else:
        coeffs = coeffmass_samples[:,:-n_masses].reshape(-1,n_masses,basis_order, n_dimensions)

    return masses, coeffs


def get_strain_from_samples(
    times, 
    recon_masses, 
    source_masses,
    recon_coeffs, 
    source_coeffs, 
    detectors=["H1"],
    return_windowed_coeffs=False, 
    window="none", 
    basis_type="chebyshev"):
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
        basis_type (str, optional): _description_. Defaults to "chebyshev".

    Returns:
        _type_: _description_
    """
    # if there is a window and I and not predicting the windowed coefficients
   
    n_masses, n_coeffs, n_dimensions = np.shape(recon_coeffs)
    if not return_windowed_coeffs and window != "none":
        n_recon_coeffs = []
        n_source_coeffs = []
        # for each mass perform the window on the xyz positions (acceleration)
        for mass in range(n_masses):
            temp_recon, win_coeffs = window_functions.perform_window(times, recon_coeffs[mass], window, basis_type=basis_type)
            n_recon_coeffs.append(temp_recon)
            if source_coeffs is not None:
                temp_source, win_coeffs = window_functions.perform_window(times, source_coeffs[mass], window, basis_type=basis_type)
                n_source_coeffs.append(temp_source)
            

        
        # update the coefficients with the windowed version
        recon_coeffs = np.array(n_recon_coeffs)
        if source_coeffs is not None:
            source_coeffs = np.array(n_source_coeffs)


    recon_strain, recon_energy = compute_waveform.get_waveform(
        times, 
        recon_masses, 
        recon_coeffs, 
        detectors, 
        basis_type=basis_type,
        compute_energy=True
    )

    source_strain, source_energy = None, None
    if source_coeffs is not None:
        source_strain, source_energy = compute_waveform.get_waveform(
            times, 
            source_masses, 
            source_coeffs, 
            detectors, 
            basis_type=basis_type,
            compute_energy=True
        )

    return recon_strain, source_strain, recon_energy, source_energy, recon_coeffs, source_coeffs

