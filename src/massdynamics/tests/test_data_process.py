from massdynamics.data_generation import data_processing
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

if __name__ == "__main__":

    test_complex_real()

    test_conversion_reverse()