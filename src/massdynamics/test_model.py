import torch
from massdynamics.data_generation import (
    data_generation,
    data_processing,
    compute_waveform,
)

from massdynamics.plotting import plotting, make_animations
from mass_dynamics.basis_functions import basis



def run_testing(config:dict) -> None:
    """ run testing (loads saved model and runs testing scripts)

    Args:
        config (dict): _description_
    """
    pre_model, model = load_models(config, config["device"])

    times, labels, strain, cshape, positions, all_dynamics = data_generation.generate_data(
        config["n_test_data"], 
        config["basis_order"], 
        config["n_masses"], 
        config["sample_rate"], 
        n_dimensions=config["n_dimensions"], 
        detectors=config["detectors"], 
        window=config["window"], 
        return_windowed_coeffs=config["return_windowed_coeffs"],
        basis_type = config["basis_type"],
        data_type = config["data_type"],
        fourier_weight=config["fourier_weight"]
        )

    try:
        strain, norm_factor = data_generation.normalise_data(strain, pre_model.norm_factor)
    except:
        print("WARNING: Normalising to different value")
        strain, norm_factor = data_generation.normalise_data(strain, None)

    """
    print(labels)
    t_mass, t_coeff = samples_to_positions_masses(
                torch.from_numpy(labels[:1]), 
                config["n_masses"],
                config["basis_order"]+2,
                config["n_dimensions"],
                config["basis_type"])

    print(np.min(t_coeff), np.max(t_coeff))
 
    source_coeffs, source_masses, source_tseries = get_time_dynamics(
                t_coeff[0],
                t_mass[0], 
                times, 
                config["n_masses"], 
                config["basis_order"]+2, 
                config["n_dimensions"], 
                basis_type=config["basis_type"])


    fig, ax = plt.subplots(nrows=3)
    ax[0].plot(positions[0, :, 0].T, color="k", label="truth")
    ax[1].plot(positions[0, :, 1].T, color="k")
    ax[2].plot(positions[0, :, 2].T, color="k")

    ax[0].plot(source_tseries[:, 0].T, ls="--", color="r", label="remake")
    ax[1].plot(source_tseries[:, 1].T, ls="--", color="r")
    ax[2].plot(source_tseries[:, 2].T, ls="--", color="r")

    fig.savefig(os.path.join(config["root_dir"], "test_pos0.png"))
    
    sys.exit()
    """
    acc_basis_order = config["basis_order"]#cshape

    n_features = acc_basis_order*config["n_masses"]*config["n_dimensions"] + config["n_masses"]

    n_context = config["sample_rate"]*2

    dataset = TensorDataset(torch.from_numpy(labels).to(torch.float32), torch.Tensor(strain))
    test_loader = DataLoader(dataset, batch_size=1)


    if config["n_dimensions"] == 1:
        test_model_1d(
            model, 
            test_loader, 
            times, 
            config["n_masses"], 
            config["basis_order"], 
            config["n_dimensions"], 
            config["root_dir"], 
            config["device"],)
    elif config["n_dimensions"] == 2:
        test_model_2d(
            model, 
            pre_model, 
            test_loader, 
            times, 
            config["n_masses"], 
            config["basis_order"], 
            config["n_dimensions"], 
            config["root_dir"], 
            config["device"])
    elif config["n_dimensions"] == 3:
        test_model_3d(
            model, 
            pre_model, 
            test_loader, 
            times, 
            config["n_masses"], 
            acc_basis_order, 
            config["n_dimensions"], 
            config["detectors"], 
            config["window"], 
            config["root_dir"], 
            config["device"], 
            config["return_windowed_coeffs"],
            basis_type=config["basis_type"])


def test_model_1d(model, dataloader, times, n_masses, basis_order, n_dimensions, root_dir, device, basis_type="chebyshev"):

    plot_out = os.path.join(root_dir, "testout")
    if not os.path.isdir(plot_out):
        os.makedirs(plot_out)

    model.eval()
    with torch.no_grad():
        for batch, (label, data) in enumerate(dataloader):
            label, data = label.to(device), data.to(device)
            coeffmass_samples = model(data.flatten(start_dim=1)).sample().cpu().numpy()

            source_coeffs, source_masses, source_tseries = get_time_dynamics(label[0].cpu().numpy(), times, n_masses, basis_order, n_dimensions, basis_type=basis_type)
            recon_coeffs, recon_masses, recon_tseries = get_time_dynamics(coeffmass_samples[0], times, n_masses, basis_order, n_dimensions, basis_type=basis_type)

            fig, ax = plt.subplots(nrows = 4)
            for mass_index in range(n_masses):
                ax[0].plot(times, source_tseries[mass_index, 0])
                ax[1].plot(times, recon_tseries[mass_index, 0])
                ax[2].plot(times, source_tseries[mass_index, 0] - recon_tseries[mass_index, 0])
    
            recon_weighted_coeffs = np.sum(recon_coeffs[:,0] * recon_masses[:, None], axis=0)
            source_weighted_coeffs = np.sum(source_coeffs[:,0] * source_masses[:, None], axis=0)

            recon_strain_coeffs = generate_strain_coefficients(recon_weighted_coeffs)
            source_strain_coeffs = generate_strain_coefficients(source_weighted_coeffs)

            recon_strain = np.polynomial.chebyshev.chebval(times, recon_strain_coeffs)
            source_strain = np.polynomial.chebyshev.chebval(times, source_strain_coeffs)


            ax[3].plot(times, recon_strain, label="recon")
            ax[3].plot(times, source_strain, label="source")
            ax[3].plot(times, data[0][0].cpu().numpy(), label="source data")

            fig.savefig(os.path.join(plot_out, f"reconstructed_{batch}.png"))


def test_model_2d(model, pre_model, dataloader, times, n_masses, basis_order, n_dimensions, root_dir, device):
    """_summary_

    Args:
        model (_type_): _description_
        pre_model (_type_): _description_
        dataloader (_type_): _description_
        times (_type_): _description_
        n_masses (_type_): _description_
        basis_order (_type_): _description_
        n_dimensions (_type_): _description_
        root_dir (_type_): _description_
        device (_type_): _description_
    """
    plot_out = os.path.join(root_dir, "testout")
    if not os.path.isdir(plot_out):
        os.makedirs(plot_out)

    model.eval()
    with torch.no_grad():
        for batch, (label, data) in enumerate(dataloader):
            label, data = label.to(device), data.to(device)
            input_data = pre_model(data)
            coeffmass_samples = model(input_data).sample().cpu().numpy()

            print(np.shape(coeffmass_samples[0]))
            source_coeffs, source_masses, source_tseries = get_time_dynamics(label[0].cpu().numpy(), times, n_masses, basis_order, n_dimensions)
            recon_coeffs, recon_masses, recon_tseries = get_time_dynamics(coeffmass_samples[0], times, n_masses, basis_order, n_dimensions)

            fig, ax = plt.subplots(nrows = 4)
            for mass_index in range(n_masses):
                ax[0].plot(times, source_tseries[mass_index,0], color="k", alpha=0.8)
                ax[0].plot(times, source_tseries[mass_index,1], color="r", alpha=0.8)
                ax[1].plot(times, recon_tseries[mass_index, 0])
                ax[1].plot(times, recon_tseries[mass_index, 1])
                ax[2].plot(times, source_tseries[mass_index,0] - recon_tseries[mass_index,0])
    
            recon_weighted_coeffs = np.sum(recon_coeffs * recon_masses[:, None, None], axis=0)
            source_weighted_coeffs = np.sum(source_coeffs * source_masses[:, None, None], axis=0)

            recon_strain_tensor = generate_2d_derivative(recon_weighted_coeffs, times)
            source_strain_tensor = generate_2d_derivative(source_weighted_coeffs, times)

            recon_strain = recon_strain_tensor[0,0] + recon_strain_tensor[0,1]
            source_strain = source_strain_tensor[0,0] + source_strain_tensor[0,1]

            ax[3].plot(times, recon_strain, label="recon")
            ax[3].plot(times, source_strain, label="source")
            ax[3].plot(times, data[0][0].cpu().numpy(), label="source data")

            fig.savefig(os.path.join(plot_out, f"reconstructed_{batch}.png"))

            make_2d_animation(plot_out, batch, recon_tseries, recon_masses, source_tseries, source_masses)


            nsamples = 50
            multi_coeffmass_samples = model(input_data).sample((nsamples, )).cpu().numpy()

            #print("multishape", multi_coeffmass_samples.shape)
            m_recon_tseries, m_recon_masses = np.zeros((nsamples, n_masses, n_dimensions, len(times))), np.zeros((nsamples, n_masses))
            for i in range(nsamples):
                #print(np.shape(multi_coeffmass_samples[i]))
                t_co, t_mass, t_time = get_time_dynamics(multi_coeffmass_samples[i][0], times, n_masses, basis_order, n_dimensions)
                m_recon_tseries[i] = t_time
                m_recon_masses[i] = t_mass

            make_2d_distribution(plot_out, batch, m_recon_tseries, m_recon_masses, source_tseries, source_masses)


def test_model_3d(
    model, 
    pre_model, 
    dataloader, 
    times, 
    n_masses, 
    basis_order, 
    n_dimensions, 
    detectors, 
    window, 
    root_dir, 
    device, 
    return_windowed_coeffs=True, 
    basis_type="chebyshev"):
    """test a 3d model sampling from the flow and producing possible trajectories

        makes animations and plots comparing models

    Args:
        model (_type_): _description_
        pre_model (_type_): _description_
        dataloader (_type_): _description_
        times (_type_): _description_
        n_masses (_type_): _description_
        basis_order (_type_): _description_
        n_dimensions (_type_): _description_
        root_dir (_type_): _description_
        device (_type_): _description_
    """
    plot_out = os.path.join(root_dir, "testout_2")
    if not os.path.isdir(plot_out):
        os.makedirs(plot_out)

    model.eval()
    with torch.no_grad():
        for batch, (label, data) in enumerate(dataloader):
            label, data = label.to(device), data.to(device)
            input_data = pre_model(data)
          
            coeffmass_samples = model(input_data).sample().cpu()

            mass_samples, coeff_samples = data_processing.samples_to_positions_masses(
                coeffmass_samples, 
                n_masses,
                basis_order,
                n_dimensions,
                basis_type)
            #print("coeffsamp", np.shape(coeff_samples))
            #print(coeff_samples[0, 0, :, 0])
            #print(coeff_samples[0, 1, :, 0])

            t_mass, t_coeff = data_processing.samples_to_positions_masses(
                label[:1].cpu(), 
                n_masses,
                basis_order,
                n_dimensions,
                basis_type)

            #print(np.shape(label), np.shape(coeffmass_samples))
            #print(np.shape(coeff_samples), np.shape(t_coeff))
            source_coeffs = t_coeff[0]
            source_masses = t_mass[0]
            recon_coeffs = coeff_samples[0]
            recon_masses = mass_samples[0]
           
            source_tseries = compute_waveform.get_time_dynamics(
                source_coeffs,
                times,  
                basis_type=basis_type)

            recon_tseries = compute_waveform.get_time_dynamics(
                recon_coeffs, 
                times,  
                basis_type=basis_type)

            """
            fig, ax = plt.subplots()
            ax.plot(source_tseries[0][0], color="k", label="truth")
            #ax.plot(recon_tseries[0][0], ls="--", color="r", label="remake")
            fig.savefig(os.path.join(root_dir, "test_pos2.png"))
            """
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

            recon_strain, _ = data_processing.normalise_data(recon_strain, pre_model.norm_factor)
            source_strain, _ = data_processing.normalise_data(source_strain, pre_model.norm_factor)

            #window = signal.windows.tukey(np.shape(source_strain)[-1], alpha=0.5)
            #recon_strain = recon_strain * window[None, :]

            fig = plotting.plot_reconstructions(
                            times, 
                            detectors, 
                            recon_strain, 
                            source_strain, 
                            data[0].cpu().numpy(), 
                            source_energy,
                            recon_energy,
                            fname = os.path.join(plot_out, f"reconstructed_{batch}.png"))

            plotting.plot_positions(
                times, 
                source_tseries, 
                recon_tseries, 
                n_dimensions, 
                n_masses,
                fname = os.path.join(plot_out, f"positions_{batch}.png"))

            plotting.plot_z_projection(
                source_tseries, 
                recon_tseries, 
                fname = os.path.join(plot_out,f"z_projection_{batch}.png"))

            """
            make_animations.make_3d_animation(
                plot_out, 
                batch, 
                recon_tseries, 
                recon_masses, 
                source_tseries, 
                source_masses)
            """

            nsamples = 500
            n_animate_samples = 50
            multi_coeffmass_samples = model(input_data).sample((nsamples, )).cpu()

            print(multi_coeffmass_samples.shape)
            multi_mass_samples, multi_coeff_samples = samples_to_positions_masses(
                multi_coeffmass_samples[:,0], 
                n_masses,
                basis_order,
                n_dimensions,
                basis_type)

            #print("multishape", multi_coeffmass_samples.shape)
            m_recon_masses = np.zeros((nsamples, n_masses))
            m_recon_tseries = np.zeros((nsamples, n_masses, n_dimensions, len(times)))
            m_recon_strain = np.zeros((nsamples, len(config["detectors"]), len(times)))
            #m_recon_energy = np.zeros((nsamples, len(times)))
            for i in range(nsamples):
                #print(np.shape(multi_coeffmass_samples[i]))
                t_co, t_mass = multi_coeff_samples[i], multi_mass_samples[i]
                t_time = compute_waveform.get_time_dynamics(
                    t_co, 
                    times, 
                    basis_type=basis_type)
                m_recon_tseries[i] = t_time
                m_recon_masses[i] = t_mass

                temp_recon_strain, temp_recon_energy, _, _, temp_m_recon_coeffs, _ = data_processing.get_strain_from_samples(
                    times, 
                    t_mass,
                    None,
                    t_co, 
                    None, 
                    detectors=["H1","L1","V1"],
                    return_windowed_coeffs=config["return_windowed_coeffs"], 
                    window=config["window"], 
                    basis_type=config["basis_type"])

                temp_recon_strain, _ = data_proccessing.normalise_data(temp_recon_strain, pre_model.norm_factor)

                m_recon_strain[i] = temp_recon_strain
                #m_recon_energy[i] = temp_recon_energy
            
            if n_masses == 2:
                print(np.shape(m_recon_masses))
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
            

            plotting.plot_sample_separations(
                times, 
                source_tseries, 
                m_recon_tseries, 
                fname=os.path.join(plot_out,f"separations_{batch}.png"))

            print("source_Strain", np.shape(source_strain))
            plotting.plot_sampled_reconstructions(
                times, 
                config["detectors"], 
                m_recon_strain, 
                source_strain, 
                fname = os.path.join(plot_out,f"recon_strain_dist_{batch}.png"))

            plotting.plot_mass_distributions(
                m_recon_masses,
                source_masses,
                fname=os.path.join(plot_out,f"massdistributions_{batch}.png"))

            
            make_animations.line_of_sight_animation(
                m_recon_tseries, 
                m_recon_masses, 
                source_tseries, 
                source_masses, 
                os.path.join(plot_out,f"2d_massdist_{batch}.gif"))


            make_animations.make_3d_distribution(
                plot_out, 
                batch, 
                m_recon_tseries[:n_animate_samples], 
                m_recon_masses[:n_animate_samples], 
                source_tseries, 
                source_masses)

            make_animations.make_3d_distribution_zproj(
                plot_out, 
                batch, 
                m_recon_tseries, 
                m_recon_masses, 
                source_tseries, 
                source_masses)