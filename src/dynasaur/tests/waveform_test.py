from dynasaur.data_generation import (
    data_generation,
    compute_waveform,
    data_processing
)

from dynasaur.basis_functions import basis

def test_fixed_velocity(output_dir):

    nsamples = 128
    times = np.linspace(-1,1,nsamples)

    xposition = np.linspace(-1,1,nsamples)
    yzposition = np.zeros(nsamples)

    position = np.stack([xposition, yzposition, yzposition])

    basis_positions = basis["fourier"]["fit"](
        times, 
        positions[None, :]
    )
    masses = np.array([0.5])

    waveform = compute_waveform.get_waveform(
        times, 
        norm_masses, 
        basis_positions, 
        ["H1"], 
        basis_type="fourier",
        compute_energy=False)

    fig, ax = plt.subplots()
    ax.plot(times, waveform)

    fig.savefig(os.path.join(output_dir, "test_waveform_fixed_vel.png"))

    