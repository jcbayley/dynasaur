from dynasaur.data_generation import data_processing, data_generation
import dynasaur
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

def test_split_order():
    n_data = 2
    n_masses = 2
    n_dimensions = 3
    n_detectors = 3
    n_times = 16

    basis_dynamics = np.arange(n_data*n_masses*n_dimensions*n_times).reshape(n_data, n_masses, n_dimensions, n_times)
    strain = np.arange(n_data*n_detectors*n_times).reshape(n_data, n_detectors, n_times)
    masses = np.arange(n_data*n_masses).reshape(n_data, n_masses)

    split_batch_times, split_strain, split_masses, split_previous_positions, split_dynamics, split_velocities, split_accelerations = dynasaur.data_generation.data_processing.split_time_data(
        basis_dynamics, 
        strain, masses, 
        None, 
        None, 
        None)

    #basis_dynamics [n_data, n_masses, n_dimensions, n_times]
    #split_dynamics [n_data*n_times, n_masses, n_dimensions]

    # check strain and masses for the 1st time of the 1st data
    assert np.all(split_strain[0] == strain[0])
    assert np.all(split_masses[0] == masses[0])
    assert np.all(split_dynamics[0, 0] == basis_dynamics[0,0,:,0])

    # check strain and masses for the last time of the 1st data
    assert np.all(split_strain[n_times - 1] == strain[0])
    assert np.all(split_masses[n_times - 1] == masses[0])
    assert np.all(split_dynamics[n_times-1, 0] == basis_dynamics[0,0,:,n_times-1])
    assert np.all(split_dynamics[:n_times] == np.transpose(basis_dynamics[0,:,:,:n_times], (2,0,1)))

    # check strain and masses for the 1st time of the 2nd data
    assert np.all(split_strain[n_times + 1] == strain[1])
    assert np.all(split_masses[n_times + 1] == masses[1])
    assert np.all(split_dynamics[n_times, 0] == basis_dynamics[1,0,:,0])

    # check batch times repeat in same way
    assert np.all(split_batch_times[:n_times] == split_batch_times[n_times:2*n_times])


def test_split_opt(return_accelerations=True, return_velocities=True, n_previous_positions=None):

    n_data = 5
    n_masses = 2
    n_dimensions = 3
    n_detectors = 3
    n_times = 16

    basis_dynamics = np.arange(n_data*n_masses*n_dimensions*n_times).reshape(n_data, n_masses, n_dimensions, n_times)
    strain = np.arange(n_data*n_detectors*n_times).reshape(n_data, n_detectors, n_times)
    masses = np.arange(n_data*n_masses).reshape(n_data, n_masses)
    if return_velocities:
        basis_velocities = np.arange(n_data*n_masses*n_dimensions*n_times).reshape(n_data, n_masses, n_dimensions, n_times)
    else:
        basis_velocities = None
    if return_accelerations:
        basis_accelerations = np.arange(n_data*n_masses*n_dimensions*n_times).reshape(n_data, n_masses, n_dimensions, n_times)
    else:
        basis_accelerations = None

    split_batch_times, split_strain, split_masses, split_previous_positions, split_dynamics, split_velocities, split_accelerations = dynasaur.data_generation.data_processing.split_time_data(
        basis_dynamics, 
        strain, masses, 
        n_previous_positions, 
        basis_velocities, 
        basis_accelerations)

    split_labels = dynasaur.data_generation.data_processing.positions_masses_to_samples(
        split_dynamics,
        split_masses,
        basis_type = "timeseries",
        velocities = split_velocities,
        accelerations = split_accelerations
        )

    unsplit_strain, unsplit_masses, unsplit_basis_dynamics, unsplit_basis_velocities, unsplit_basis_accelerations = dynasaur.data_generation.data_processing.unsplit_time_data(
        split_labels, 
        split_strain, 
        n_masses,
        n_dimensions,
        n_times,
        return_accelerations, 
        return_velocities)

    assert np.all(unsplit_strain == strain)
    assert np.all(unsplit_masses == masses)
    assert np.all(unsplit_basis_dynamics == basis_dynamics)
    if return_velocities:
        assert np.all(unsplit_basis_velocities == basis_velocities)
    if return_accelerations:
        assert np.all(unsplit_basis_accelerations == basis_accelerations)

def test_split():
    test_split_opt(return_accelerations=True, return_velocities=True, n_previous_positions=None)
    test_split_opt(return_accelerations=True, return_velocities=False, n_previous_positions=None)
    test_split_opt(return_accelerations=False, return_velocities=True, n_previous_positions=None)
    test_split_opt(return_accelerations=False, return_velocities=False, n_previous_positions=None)
    test_split_opt(return_accelerations=False, return_velocities=False, n_previous_positions=1)

def reshape_test():

    batch, n_mass, n_dim, n_time = 10,2,3,10
    input_data = np.arange(batch*n_mass*n_dim*n_time).reshape(batch, n_mass, n_dim, n_time)
    
    reshape_input = reshape_to_time_batch(input_data)

    reshape_output = reshape_to_original(reshape_input, batch, n_time)

    assert reshape_output == input_data


def preprocess_full_test():

    n_data = 5
    n_masses = 2
    n_dimensions = 3
    n_detectors = 3
    n_times = 16
    return_accelerations=True
    return_velocities=True

    basis_dynamics = np.arange(n_data*n_masses*n_dimensions*n_times).reshape(n_data, n_masses, n_dimensions, n_times)
    strain = np.arange(n_data*n_detectors*n_times).reshape(n_data, n_detectors, n_times)
    masses = np.arange(n_data*n_masses).reshape(n_data, n_masses)
    n_previous_positions = None
    basis_velocities = np.arange(n_data*n_masses*n_dimensions*n_times).reshape(n_data, n_masses, n_dimensions, n_times)
    basis_accelerations = np.arange(n_data*n_masses*n_dimensions*n_times).reshape(n_data, n_masses, n_dimensions, n_time)

    pre_model, labels, strain, batch_times, previous_positions = preprocess_data(
        pre_model, 
        basis_dynamics,
        masses, 
        strain, 
        basis_velocities=basis_velocities,
        basis_accelerations=basis_accelerations,
        window_strain=None, 
        spherical_coords=None, 
        initial_run=False,
        n_masses=2,
        device="cpu",
        basis_type="fourier",
        n_dimensions=3,
        split_data=False,
        n_previous_positions="none")

if __name__ == "__main__":

    test_complex_real()

    test_conversion_reverse()