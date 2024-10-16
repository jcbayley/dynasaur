from dynasaur.data_generation import data_processing
import numpy as np
import torch

def test_conversion():

    n_samples = 1
    n_masses = 2
    n_dimensions = 3
    n_coeffs = 16
    basis_type = "fourier"

    coeffmass_samples = torch.from_numpy(
        np.random.uniform(0,1,size = (n_samples, n_masses*n_dimensions*int(n_coeffs*0.5 +1)*2 + n_masses))
    )

    masses, coeffs = data_processing.samples_to_positions_masses(
        coeffmass_samples, 
        n_masses,
        n_coeffs,
        n_dimensions,
        basis_type)

    samples_remake = data_processing.positions_masses_to_samples(
        coeff_samples = coeffs,
        mass_samples = masses,
        basis_type = basis_type
        )

    print(samples_remake - coeffmass_samples.numpy())

def test_conversion_reverse():

    n_samples = 1
    n_masses = 2
    n_dimensions = 3
    n_coeffs = 16
    basis_type = "fourier"

    coeff_samples = np.random.uniform(-1,1,size = (n_samples, n_masses,n_dimensions,int(n_coeffs*0.5 +1))) +1j*np.random.uniform(-1,1,size = (n_samples, n_masses,n_dimensions,int(n_coeffs*0.5 +1)) )
    

    mass_samples = np.random.uniform(0,1, size=(n_samples, n_masses))
    """
    print(np.shape(coeff_samples))
    output_coeffs1 = data_processing.complex_to_real(coeff_samples)#.reshape(n_samples, -1)
    print(np.shape(output_coeffs1))
    output_coeffs2 = output_coeffs1.reshape(n_samples, -1)
    output_coeffs = np.concatenate([output_coeffs2, torch.from_numpy(mass_samples)], axis=1)

    masses = output_coeffs[:, -n_masses:]

    print("mass")
    print(masses - mass_samples)

    sshape = np.shape(output_coeffs[:, :-n_masses])

    coeffs = output_coeffs[:, :-n_masses].reshape(
            sshape[0], 
            n_masses, 
            n_dimensions, 
            int(n_coeffs/2 + 1), 
            2)
    print(np.shape(coeffs), np.shape(output_coeffs1))
    print(coeffs - output_coeffs1)

    coeffs = data_processing.real_to_complex(coeffs)
    print(coeffs - coeff_samples)
    """

    samples_remake = data_processing.positions_masses_to_samples(
        coeff_samples = coeff_samples,
        mass_samples = mass_samples,
        basis_type = basis_type
        )

    masses, coeffs = data_processing.samples_to_positions_masses(
        samples_remake, 
        n_masses,
        n_coeffs,
        n_dimensions,
        basis_type)


    print(coeffs - coeff_samples)
    
def test_complex_real():


    samples = np.random.uniform(0,1,size = (32,2))

    csamps = data_processing.real_to_complex(
        samples)

    rsamps = data_processing.complex_to_real(
        csamps)


    print(samples - rsamps)

def test_strain_reconstruct():

    n_data = 1
    n_masses = 2
    n_dimensions = 3
    basis_order = 16
    sample_rate = 128
    basis_type = "fourier"
    detectors = ["H1"]

    times, positions, masses, position_coeffs = kepler_orbits.generate_data(
                n_data,
                detectors = detectors,
                n_masses=n_masses,
                basis_order = basis_order,
                basis_type = basis_type,
                n_dimensions = n_dimensions,
                sample_rate = sample_rate)

    strain_timeseries, energy = compute_waveform.get_waveform(
            times, 
            masses[0], 
            all_basis_dynamics[0], 
            detectors, 
            basis_type=basis_type,
            compute_energy=False)

    recon_strain, source_strain, recon_energy, source_energy, recon_coeffs, source_coeffs = data_processing.get_strain_from_samples(
                times, 
                recon_masses,
                source_masses, 
                recon_coeffs, 
                source_coeffs, 
                detectors=detectors,
                return_windowed_coeffs=return_windowed_coeffs, 
                window=window, 
                basis_type=basis_type)


    t_mass, t_coeff = data_processing.samples_to_positions_masses(
                label[:1].cpu().numpy(), 
                n_masses,
                basis_order,
                n_dimensions,
                basis_type)

    #print(np.shape(label), np.shape(coeffmass_samples))
    #print(np.shape(coeff_samples), np.shape(t_coeff))
    source_coeffs = t_coeff[0]
    source_masses = t_mass[0]
    
    source_strain, source_energy,source_coeffs = data_processing.get_strain_from_samples(
        times, 
        source_masses,  
        source_coeffs, 
        detectors=detectors,
        return_windowed_coeffs=return_windowed_coeffs, 
        window=window, 
        basis_type=basis_type)


    source_strain, _ = data_processing.normalise_data(source_strain, pre_model.norm_factor)

if __name__ == "__main__":

    test_complex_real()

    test_conversion_reverse()