import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import os
import numpy as np
import copy
from dynasaur.data_generation import (
    data_generation,
    data_processing,
    compute_waveform
)
from dynasaur.create_model import (
    load_models,
    create_models
)

from dynasaur.plotting import plotting, make_animations
from dynasaur.basis_functions import basis
import h5py
import matplotlib.pyplot as plt


def run_testing(config:dict, make_plots=False, n_test=None) -> None:
    """ run testing (loads saved model and runs testing scripts)

    Args:
        config (dict): _description_
    """
    pre_model, model, weights = load_models(config, config["device"])

    config.setdefault("coordinate_type", "cartesian")


    n_test = config["n_test_data"] if n_test is None else n_test

    times, basis_dynamics, masses, strain, cshape, positions, all_dynamics, snr = data_generation.generate_data(
        n_test, 
        config["basis_order"], 
        config["n_masses"], 
        config["sample_rate"], 
        n_dimensions=3, 
        detectors=config["detectors"], 
        window=config["window"], 
        window_acceleration=config["window_acceleration"],
        basis_type = config["basis_type"],
        data_type = config["data_type"],
        fourier_weight=config["fourier_weight"],
        coordinate_type=config["coordinate_type"],
        noise_variance=config["noise_variance"],
        snr=config["snr"],
        prior_args=config["prior_args"]
        )


    pre_model, labels, strain = data_processing.preprocess_data(
        pre_model, 
        basis_dynamics,
        masses, 
        strain, 
        window_strain=config["window_strain"], 
        spherical_coords=config["spherical_coords"], 
        initial_run=False,
        n_masses=config["n_masses"],
        device=config["device"],
        basis_type=config["basis_type"],
        n_dimensions=config["n_dimensions"])

    """
    strain, norm_factor = data_processing.normalise_data(
        strain, 
        pre_model.norm_factor)
    labels, label_norm_factor, mass_norm_factor = data_processing.normalise_labels(
        labels, 
        pre_model.label_norm_factor, 
        pre_model.mass_norm_factor, 
        n_masses=config["n_masses"])
    """

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


    upsample_times = np.linspace(np.min(times), np.max(times), config["plot_sample_rate"])


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
            model=model, 
            pre_model=pre_model, 
            dataloader=test_loader,
            times=times,
            upsample_times=upsample_times, 
            n_masses=config["n_masses"], 
            basis_order=acc_basis_order, 
            n_dimensions=config["n_dimensions"], 
            detectors=config["detectors"], 
            window=config["window"], 
            root_dir=config["root_dir"], 
            device=config["device"], 
            window_acceleration=config["window_acceleration"],
            window_strain=config["window_strain"],
            spherical_coords=config["spherical_coords"],
            basis_type=config["basis_type"],
            sky_position=config["prior_args"]["sky_position"],
            make_plots=make_plots,
            flow_package=config["flow_model_type"].split("-")[0])
    elif config["n_dimensions"] == 3:
        test_model_3d(
            model=model, 
            pre_model=pre_model, 
            dataloader=test_loader,
            times=times,
            upsample_times=upsample_times, 
            n_masses=config["n_masses"], 
            basis_order=acc_basis_order, 
            n_dimensions=config["n_dimensions"], 
            detectors=config["detectors"], 
            window=config["window"], 
            root_dir=config["root_dir"], 
            device=config["device"], 
            window_acceleration=config["window_acceleration"],
            window_strain=config["window_strain"],
            spherical_coords=config["spherical_coords"],
            basis_type=config["basis_type"],
            sky_position=config["prior_args"]["sky_position"],
            make_plots=make_plots,
            flow_package=config["flow_model_type"].split("-")[0])



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


def test_model_2d(
    model, 
    pre_model, 
    dataloader, 
    times, 
    upsample_times,
    n_masses, 
    basis_order, 
    n_dimensions, 
    detectors, 
    window, 
    root_dir, 
    device, 
    n_samples=2000,
    n_animate_samples=50,
    window_acceleration=True, 
    basis_type="chebyshev",
    window_strain=None,
    spherical_coords=False,
    make_plots=True,
    sky_position=(np.pi, np.pi/2),
    flow_package="zuko"):
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

    data_out = os.path.join(plot_out, "data_output")
    if not os.path.isdir(data_out):
        os.makedirs(data_out)

    n_detectors = len(detectors)
    model.eval()
    with torch.no_grad():
        for batch, (label, data) in enumerate(dataloader):
            label, data = label.to(device), data.to(device)
            input_data = pre_model(data)
            if flow_package == "zuko":
                coeffmass_samples = model(input_data).sample().cpu().numpy()
            elif flow_package == "glasflow":
                coeffmass_samples = model.sample(1, conditional=input_data).cpu().numpy()
            else:
                raise Exception(f"No flow package {flow_package}")
            print("ccoeff1", np.max(coeffmass_samples[:,:-2]))
            print("cmass1", np.max(coeffmass_samples[:,-2:]))
            pre_model, mass_samples, coeff_samples, _ = data_processing.unpreprocess_data(
                pre_model, 
                coeffmass_samples, 
                data.cpu().numpy(), 
                window_strain=window_strain, 
                spherical_coords=spherical_coords, 
                initial_run=False,
                n_masses=n_masses,
                n_dimensions=n_dimensions,
                device=device,
                basis_type=basis_type,
                basis_order=basis_order)
            print("ccoeff2",np.max(coeff_samples))
            print("cmass2", np.max(mass_samples))
            print("csshape:", np.shape(coeff_samples))

            _, t_mass, t_coeff, _ = data_processing.unpreprocess_data(
                pre_model, 
                label.cpu().numpy(), 
                data.cpu().numpy(), 
                window_strain=window_strain, 
                spherical_coords=spherical_coords, 
                initial_run=False,
                n_masses=n_masses,
                n_dimensions=n_dimensions,
                device=device,
                basis_type=basis_type,
                basis_order=basis_order)
            """
            #label = label.cpu().numpy()
            label, nf, mf = data_processing.unnormalise_labels(
                label.cpu().numpy(), 
                pre_model.label_norm_factor, 
                pre_model.mass_norm_factor,
                n_masses=n_masses)


            t_mass, t_coeff = data_processing.samples_to_positions_masses(
                label[:1], 
                n_masses,
                basis_order,
                n_dimensions,
                basis_type)
            """
            #print(np.shape(label), np.shape(coeffmass_samples))
            #print(np.shape(coeff_samples), np.shape(t_coeff))
            source_coeffs = t_coeff[0]
            source_masses = t_mass[0]
            recon_coeffs = coeff_samples[0]
            recon_masses = mass_samples[0]

            source_tseries = compute_waveform.get_time_dynamics(
                source_coeffs,
                upsample_times,  
                basis_type=basis_type)

            recon_tseries = compute_waveform.get_time_dynamics(
                recon_coeffs, 
                upsample_times,  
                basis_type=basis_type)

            recon_strain, recon_energy, recon_coeffs = data_processing.get_strain_from_samples(
                upsample_times, 
                recon_masses,
                recon_coeffs, 
                detectors=detectors,
                window_acceleration=window_acceleration, 
                window=window, 
                basis_type=basis_type,
                basis_order=basis_order,
                sky_position=sky_position)
            
            source_strain, source_energy,source_coeffs = data_processing.get_strain_from_samples(
                upsample_times, 
                source_masses,  
                source_coeffs, 
                detectors=detectors,
                window_acceleration=window_acceleration, 
                window=window, 
                basis_type=basis_type,
                basis_order=basis_order,
                sky_position=sky_position)
            
            
            # preprocess the strain again
            #print("masses", source_masses, recon_masses)
            #print("coeffs", source_coeffs, recon_coeffs)
            #print("strain", source_strain, recon_strain)
            #print("rstrain1", np.max(recon_strain))
            _, _, recon_strain = data_processing.preprocess_data(
                pre_model, 
                coeff_samples, 
                mass_samples,
                recon_strain, 
                window_strain=window_strain, 
                spherical_coords=spherical_coords, 
                initial_run=False,
                n_masses=n_masses,
                device=device)
            #print("rstrain2", np.max(recon_strain))

            #print("sstrain1", np.max(source_strain))
            _, _, source_strain = data_processing.preprocess_data(
                pre_model, 
                coeff_samples, 
                mass_samples,
                source_strain, 
                window_strain=window_strain, 
                spherical_coords=spherical_coords, 
                initial_run=False,
                n_masses=n_masses,
                device=device)              
            print("norms", pre_model.norm_factor, pre_model.label_norm_factor)
            print("vals", np.max(coeff_samples), np.max(source_coeffs))
            #recon_strain, _ = data_processing.normalise_data(recon_strain, pre_model.norm_factor)
            #source_strain, _ = data_processing.normalise_data(source_strain, pre_model.norm_factor)
            source_plot_data = data[0].cpu().numpy()
            #print("sstrain2", np.max(source_strain), np.max(source_plot_data), np.max(recon_strain))
            #window = signal.windows.tukey(np.shape(source_strain)[-1], alpha=0.5)
            #recon_strain = recon_strain * window[None, :]
            if make_plots:
                fig = plotting.plot_reconstructions(
                                upsample_times, 
                                detectors, 
                                recon_strain, 
                                source_strain, 
                                source_plot_data, 
                                source_energy,
                                recon_energy,
                                fname = os.path.join(plot_out, f"reconstructed_{batch}.png"))


            if flow_package == "zuko":
                multi_coeffmass_samples = model(input_data).sample((n_samples, )).cpu().numpy()
            elif flow_package == "glasflow":
                multi_coeffmass_samples = model.sample(n_samples, conditional=input_data).cpu().numpy()
            else:
                raise Exception(f"No flow package {flow_package}")

    
            pre_model, multi_mass_samples, multi_coeff_samples, _ = data_processing.unpreprocess_data(
                pre_model, 
                multi_coeffmass_samples[:,0], 
                data.cpu().numpy(), 
                window_strain=window_strain, 
                spherical_coords=spherical_coords, 
                initial_run=False,
                n_masses=n_masses,
                n_dimensions=n_dimensions,
                device=device,
                basis_type=basis_type,
                basis_order=basis_order)

            print("mcshape", np.shape(multi_coeff_samples))

            n_dimensions_out = 3
            #print("multishape", multi_coeffmass_samples.shape)
            m_recon_masses = np.zeros((n_samples, n_masses))
            m_recon_tseries = np.zeros((n_samples, n_masses, n_dimensions_out, len(upsample_times)))
            m_recon_tseries = np.zeros((n_samples, n_masses, n_dimensions_out, len(upsample_times)))

            m_recon_strain = np.zeros((n_samples, len(detectors), len(upsample_times)))
            m_recon_strain_coeffs = np.zeros((n_samples, len(detectors), int(0.5*len(upsample_times))))
            #m_recon_energy = np.zeros((nsamples, len(times)))

            #multi_coeff_samples[:,:,2] = 0
            #multi_coeff_samples[:,0] = 0

            for i in range(n_samples):
                #print(np.shape(multi_coeffmass_samples[i]))
                t_co, t_mass = multi_coeff_samples[i], multi_mass_samples[i]

                t_time = compute_waveform.get_time_dynamics(
                    t_co, 
                    upsample_times, 
                    basis_type=basis_type)

                m_recon_tseries[i] = t_time
                m_recon_masses[i] = t_mass
                #print(np.min(t_co), np.max(t_co), t_mass)
                temp_recon_strain, temp_recon_energy, temp_m_recon_coeffs = data_processing.get_strain_from_samples(
                    upsample_times, 
                    t_mass,
                    t_co,  
                    detectors=detectors,
                    window_acceleration=window_acceleration, 
                    window=window, 
                    basis_type=basis_type,
                    basis_order=basis_order,
                    sky_position=sky_position)

                _, _, temp_recon_strain = data_processing.preprocess_data(
                    pre_model, 
                    coeff_samples, 
                    mass_samples,
                    temp_recon_strain, 
                    window_strain=window_strain, 
                    spherical_coords=spherical_coords, 
                    initial_run=False,
                    n_masses=n_masses,
                    device=device)
                
                #temp_recon_strain, _ = data_processing.normalise_data(temp_recon_strain, pre_model.norm_factor)

                m_recon_strain[i] = temp_recon_strain
                #m_recon_energy[i] = temp_recon_energy

            fig, ax = plt.subplots(ncols=n_detectors, nrows=3, figsize=(9,7))
            for pi in range(n_detectors):
                for mi in range(2):
                    ax[mi,pi].boxplot(np.abs(multi_coeff_samples[:,mi,pi]), showfliers=False)
                    ax[mi,pi].plot(np.arange(len(source_coeffs[mi][pi])) + 1, np.abs(source_coeffs[mi][pi]), label="source", color="C0", ls="--")
     
            ax[2,0].hist(np.log(multi_mass_samples[:,0]), bins=100)
            ax[2,1].hist(np.log(multi_mass_samples[:,1]), bins=100)
            ax[2,0].axvline(np.log(source_masses[0]), color="r")
            ax[2,1].axvline(np.log(source_masses[1]), color="r")
            msun=1.0e30
            lmsun = np.log(msun)
            ax[2,1].set_xlabel("Mass 2")
            ax[2,0].set_xlabel("Mass 1")
            ax[0,0].set_ylabel("Mass 2")
            ax[1,0].set_ylabel("Mass 1")
            ax[1,0].set_xlabel("x dimension power")
            ax[1,1].set_xlabel("y dimension power")
            #ax[2,1].set_xlim([1e-6*msun, 7e-3*msun])
            #ax[2,0].set_xlim([1.8*msun, 2.2*msun])
            #ax[2,1].set_xlim([lmsun-4, lmsun])
            #ax[2,0].set_xlim([lmsun-1, lmsun+1])
            #ax[0,0].legend()
            fig.tight_layout()
            fig.savefig(os.path.join(plot_out, f"test_power_{batch}.png"))

            fig, ax = plt.subplots(ncols=n_detectors*2, nrows=3, figsize=(9,7))
            for pi in range(n_detectors):
                for mi in range(2):
                    t_re = np.real(multi_coeff_samples[:,mi,pi])
                    t_im = np.imag(multi_coeff_samples[:,mi,pi])
                    s_re = np.real(source_coeffs[mi][pi])
                    s_im = np.imag(source_coeffs[mi][pi])
                    ax[mi,pi].violinplot(t_re, widths=1.0)
                    ax[mi,pi+n_detectors].violinplot(t_im,widths=1.0)
                    ax[mi,pi].plot(np.arange(len(source_coeffs[mi][pi])) + 1, s_re, label="source real", color="k", ls="--")
                    ax[mi,pi+n_detectors].plot(np.arange(len(source_coeffs[mi][pi])) + 1, s_im, label="source imag", color="k", ls="--")

                    ax[mi,pi].set_ylim(np.min(s_re), np.max(s_re))
                    ax[mi,pi+n_detectors].set_ylim(np.min(s_im), np.max(s_im))

            ax[2,0].hist(np.log(multi_mass_samples[:,0]), bins=100)
            ax[2,1].hist(np.log(multi_mass_samples[:,1]), bins=100)
            ax[2,0].axvline(np.log(source_masses[0]), color="r")
            ax[2,1].axvline(np.log(source_masses[1]), color="r")
            msun=1.0e30
            lmsun = np.log(msun)
            ax[0,0].set_ylabel("Mass 1")
            ax[1,0].set_ylabel("Mass 2")
            ax[2,1].set_xlabel("Mass 2")
            ax[2,0].set_xlabel("Mass 1")
            ax[1,0].set_xlabel("x dimension real")
            ax[1,1].set_xlabel("x dimension imag")
            ax[1,2].set_xlabel("y dimension real")
            ax[1,3].set_xlabel("y dimension imag")
            #ax[2,1].set_xlim([1e-6*msun, 7e-3*msun])
            #ax[2,0].set_xlim([1.8*msun, 2.2*msun])
            #ax[2,1].set_xlim([lmsun-4, lmsun])
            #ax[2,0].set_xlim([lmsun-1, lmsun+1])

            #ax[0,0].legend()
            fig.tight_layout()
            fig.savefig(os.path.join(plot_out, f"test_coeffs_{batch}.png"))

            fig, ax = plt.subplots( nrows=n_detectors, figsize=(9,7))
            for pi in range(n_detectors):
                ax[pi].boxplot(m_recon_strain[:,pi], showfliers=False)
                ax[pi].plot(np.arange(len(source_strain[pi])) + 1, source_strain[pi], label="source", color="C0", ls="--")
     
            #ax[2,1].set_xlim([1e-6*msun, 7e-3*msun])
            #ax[2,0].set_xlim([1.8*msun, 2.2*msun])
            #ax[2,1].set_xlim([lmsun-4, lmsun])
            #ax[2,0].set_xlim([lmsun-1, lmsun+1])
            #ax[0,0].legend()
            fig.savefig(os.path.join(plot_out, f"test_strain_{batch}.png"))
            #print("strainmax", np.max(source_strain), np.max(m_recon_strain))
            #print("labelmax", np.max(source_coeffs), np.max(multi_coeff_samples))
            if n_masses == 2:
                #print(np.shape(m_recon_masses))
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


            #############################
            ## SAVE THE DATA
            #############################

            with h5py.File(os.path.join(data_out, f"data_{batch}.hdf5"), "w") as f:
                f.create_dataset("recon_timeseries", data=m_recon_tseries)
                f.create_dataset("recon_strain", data=m_recon_strain)
                f.create_dataset("recon_masses", data=m_recon_masses)
                f.create_dataset("source_timeseries", data=source_tseries)
                f.create_dataset("source_strain", data=data[0].cpu().numpy())
                f.create_dataset("source_strain_signal_only", data=source_strain)
                f.create_dataset("source_masses", data=source_masses)
                f.create_dataset("source_basis", data=source_coeffs)
                f.create_dataset("recon_basis", data=multi_coeff_samples)

            if make_plots:

                plotting.plot_dimension_projection(
                    m_recon_tseries[:10], 
                    source_tseries, 
                    fname=os.path.join(plot_out, f"dim_projection_{batch}.png"), 
                    alpha=0.2)

                print("source_Strain", np.shape(source_strain))
                plotting.plot_sampled_reconstructions(
                    upsample_times, 
                    detectors, 
                    m_recon_strain, 
                    source_strain, 
                    fname = os.path.join(plot_out,f"recon_strain_dist_{batch}.png"))

                plotting.plot_mass_distributions(
                    m_recon_masses,
                    source_masses,
                    fname=os.path.join(plot_out,f"massdistributions_{batch}.png"))
                

                
                make_animations.heatmap_projections(
                    m_recon_tseries, 
                    m_recon_masses, 
                    source_tseries, 
                    source_masses, 
                    os.path.join(plot_out,f"heatmap_projections_{batch}.gif"),
                    duration=5)
    


def test_model_3d(
    model, 
    pre_model, 
    dataloader, 
    times, 
    upsample_times,
    n_masses, 
    basis_order, 
    n_dimensions, 
    detectors, 
    window, 
    root_dir, 
    device, 
    n_samples=2000,
    n_animate_samples=50,
    window_acceleration=True, 
    basis_type="chebyshev",
    window_strain=None,
    spherical_coords=False,
    sky_position=(np.pi, np.pi/2),
    make_plots=True,
    flow_package="zuko"):
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

    data_out = os.path.join(plot_out, "data_output")
    if not os.path.isdir(data_out):
        os.makedirs(data_out)

    model.eval()
    with torch.no_grad():
        for batch, (label, data) in enumerate(dataloader):
            label, data = label.to(device), data.to(device)
            input_data = pre_model(data)
          
          
            if flow_package == "zuko":
                coeffmass_samples = model(input_data).sample().cpu().numpy()
            elif flow_package == "glasflow":
                coeffmass_samples = model.sample(1, conditional=input_data).cpu().numpy()
            else:
                raise Exception(f"No flow package {flow_package}")
            print("ccoeff1", np.max(coeffmass_samples[:,:-2]))
            print("cmass1", np.max(coeffmass_samples[:,-2:]))
            pre_model, mass_samples, coeff_samples, _ = data_processing.unpreprocess_data(
                pre_model, 
                coeffmass_samples, 
                data.cpu().numpy(), 
                window_strain=window_strain, 
                spherical_coords=spherical_coords, 
                initial_run=False,
                n_masses=n_masses,
                n_dimensions=n_dimensions,
                device=device,
                basis_type=basis_type,
                basis_order=basis_order)
            print("ccoeff2",np.max(coeff_samples))
            print("cmass2", np.max(mass_samples))
    

            print("bmasstr",np.min(label.cpu().numpy()[:,-n_masses:]), np.max(label.cpu().numpy()[:,-n_masses:]))
            print("bcoefftr",np.min(label.cpu().numpy()[:,:-n_masses]), np.max(label.cpu().numpy()[:,:-n_masses]))

            #label = label.cpu().numpy()
            label, nf, mf = data_processing.unnormalise_labels(
                label.cpu().numpy(), 
                pre_model.label_norm_factor, 
                pre_model.mass_norm_factor,
                n_masses=n_masses)

            print("amasstr",np.min(label[:,-n_masses:]), np.max(label[:,-n_masses:]))
            print("acoefftr",np.min(label[:,:-n_masses]), np.max(label[:,:-n_masses]))

            """
            mass_samples, coeff_samples = data_processing.samples_to_positions_masses(
                coeffmass_samples, 
                n_masses,
                basis_order,
                n_dimensions,
                basis_type)
            """
            #print("coeffsamp", np.shape(coeff_samples))
            #print(coeff_samples[0, 0, :, 0])
            #print(coeff_samples[0, 1, :, 0])

            t_mass, t_coeff = data_processing.samples_to_positions_masses(
                label[:1], 
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
                upsample_times,  
                basis_type=basis_type)

            recon_tseries = compute_waveform.get_time_dynamics(
                recon_coeffs, 
                upsample_times,  
                basis_type=basis_type)

            """
            fig, ax = plt.subplots()
            ax.plot(source_tseries[0][0], color="k", label="truth")
            #ax.plot(recon_tseries[0][0], ls="--", color="r", label="remake")
            fig.savefig(os.path.join(root_dir, "test_pos2.png")
            
            print(np.shape(recon_coeffs), np.shape(source_coeffs))
            fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(9,7))
            for pi in range(3):
                for mi in range(2):
                    ax[mi,pi].plot(np.abs(recon_coeffs[mi][pi]), label="recon", color="C0")
                    ax[mi,pi].plot(np.abs(source_coeffs[mi][pi]), label="source", color="C0", ls="--")
     
            ax[0,0].legend()
            fig.savefig(os.path.join(plot_out, f"test_power_{batch}.png"))
            """

            print("cfs", np.max(recon_coeffs), np.max(source_coeffs))
            recon_strain, recon_energy, recon_coeffs = data_processing.get_strain_from_samples(
                upsample_times, 
                recon_masses,
                recon_coeffs, 
                detectors=detectors,
                window_acceleration=window_acceleration, 
                window=window, 
                basis_type=basis_type,
                basis_order=basis_order,
                sky_position=sky_position)
            """
            source_strain, source_energy = compute_waveform.get_waveform(
                times, 
                source_masses, 
                source_coeffs, 
                detectors, 
                basis_type=basis_type,
                compute_energy=False)
            """
            
            source_strain, source_energy,source_coeffs = data_processing.get_strain_from_samples(
                upsample_times, 
                source_masses,  
                source_coeffs, 
                detectors=detectors,
                window_acceleration=window_acceleration, 
                window=window, 
                basis_type=basis_type,
                basis_order=basis_order,
                sky_position=sky_position)
            
            
            # preprocess the strain again
            #print("masses", source_masses, recon_masses)
            #print("coeffs", source_coeffs, recon_coeffs)
            #print("strain", source_strain, recon_strain)
            #print("rstrain1", np.max(recon_strain))
            _, _, recon_strain = data_processing.preprocess_data(
                pre_model, 
                coeff_samples, 
                mass_samples,
                recon_strain, 
                window_strain=window_strain, 
                spherical_coords=spherical_coords, 
                initial_run=False,
                n_masses=n_masses,
                device=device)
            #print("rstrain2", np.max(recon_strain))

            #print("sstrain1", np.max(source_strain))
            _, _, source_strain = data_processing.preprocess_data(
                pre_model, 
                coeff_samples, 
                mass_samples,
                source_strain, 
                window_strain=window_strain, 
                spherical_coords=spherical_coords, 
                initial_run=False,
                n_masses=n_masses,
                device=device)              
            print("norms", pre_model.norm_factor, pre_model.label_norm_factor)
            print("vals", np.max(coeff_samples), np.max(source_coeffs))
            #recon_strain, _ = data_processing.normalise_data(recon_strain, pre_model.norm_factor)
            #source_strain, _ = data_processing.normalise_data(source_strain, pre_model.norm_factor)
            source_plot_data = data[0].cpu().numpy()
            #print("sstrain2", np.max(source_strain), np.max(source_plot_data), np.max(recon_strain))
            #window = signal.windows.tukey(np.shape(source_strain)[-1], alpha=0.5)
            #recon_strain = recon_strain * window[None, :]
            if make_plots:
                fig = plotting.plot_reconstructions(
                                upsample_times, 
                                detectors, 
                                recon_strain, 
                                source_strain, 
                                source_plot_data, 
                                source_energy,
                                recon_energy,
                                fname = os.path.join(plot_out, f"reconstructed_{batch}.png"))

                """
                plotting.plot_positions(
                    upsample_times, 
                    source_tseries, 
                    recon_tseries, 
                    n_dimensions, 
                    n_masses,
                    fname = os.path.join(plot_out, f"positions_{batch}.png"))

                plotting.plot_z_projection(
                    source_tseries, 
                    recon_tseries, 
                    fname = os.path.join(plot_out,f"z_projection_{batch}.png"))

                
                make_animations.make_3d_animation(
                    plot_out, 
                    batch, 
                    recon_tseries, 
                    recon_masses, 
                    source_tseries, 
                    source_masses)
                """

            if flow_package == "zuko":
                multi_coeffmass_samples = model(input_data).sample((n_samples, )).cpu().numpy()
            elif flow_package == "glasflow":
                multi_coeffmass_samples = model.sample(n_samples, conditional=input_data).cpu().numpy()
            else:
                raise Exception(f"No flow package {flow_package}")

    
            pre_model, multi_mass_samples, multi_coeff_samples, _ = data_processing.unpreprocess_data(
                pre_model, 
                multi_coeffmass_samples[:,0], 
                data.cpu().numpy(), 
                window_strain=window_strain, 
                spherical_coords=spherical_coords, 
                initial_run=False,
                n_masses=n_masses,
                n_dimensions=n_dimensions,
                device=device,
                basis_type=basis_type,
                basis_order=basis_order)
            #print(np.shape(multi_coeffmass_samples))
            #print("lnorm",np.max(multi_coeffmass_samples[:,:,:-2]), pre_model.label_norm_factor)

            """
            print("b",np.min(multi_coeffmass_samples), np.max(multi_coeffmass_samples), pre_model.label_norm_factor, np.shape(multi_coeffmass_samples))
            multi_coeffmass_samples, nf, mf = data_processing.unnormalise_labels(
                multi_coeffmass_samples[:,0], 
                pre_model.label_norm_factor, 
                pre_model.mass_norm_factor,
                n_masses=n_masses)
            print("a",np.min(multi_coeffmass_samples), np.max(multi_coeffmass_samples))
            """
            
            #plotting.plot_1d_posteriors(multi_coeffmass_samples, label[0], fname=os.path.join(plot_out,f"posterior_1d_{batch}.png"))

            """
            print("mmsamples", multi_coeffmass_samples.shape, multi_coeffmass_samples.dtype)
            multi_mass_samples, multi_coeff_samples = data_processing.samples_to_positions_masses(
                multi_coeffmass_samples, 
                n_masses,
                basis_order,
                n_dimensions,
                basis_type)
            """

            #print("multishape", multi_coeffmass_samples.shape)
            m_recon_masses = np.zeros((n_samples, n_masses))
            m_recon_tseries = np.zeros((n_samples, n_masses, n_dimensions, len(upsample_times)))
            m_recon_tseries = np.zeros((n_samples, n_masses, n_dimensions, len(upsample_times)))

            m_recon_strain = np.zeros((n_samples, len(detectors), len(upsample_times)))
            m_recon_strain_coeffs = np.zeros((n_samples, len(detectors), int(0.5*len(upsample_times))))
            #m_recon_energy = np.zeros((nsamples, len(times)))

            #multi_coeff_samples[:,:,2] = 0
            #multi_coeff_samples[:,0] = 0

            for i in range(n_samples):
                #print(np.shape(multi_coeffmass_samples[i]))
                t_co, t_mass = multi_coeff_samples[i], multi_mass_samples[i]

                t_time = compute_waveform.get_time_dynamics(
                    t_co, 
                    upsample_times, 
                    basis_type=basis_type)

                m_recon_tseries[i] = t_time
                m_recon_masses[i] = t_mass
                #print(np.min(t_co), np.max(t_co), t_mass)
                temp_recon_strain, temp_recon_energy, temp_m_recon_coeffs = data_processing.get_strain_from_samples(
                    upsample_times, 
                    t_mass,
                    t_co,  
                    detectors=detectors,
                    window_acceleration=window_acceleration, 
                    window=window, 
                    basis_type=basis_type,
                    basis_order=basis_order,
                    sky_position=sky_position)

                _, _, temp_recon_strain = data_processing.preprocess_data(
                    pre_model, 
                    coeff_samples, 
                    mass_samples,
                    temp_recon_strain, 
                    window_strain=window_strain, 
                    spherical_coords=spherical_coords, 
                    initial_run=False,
                    n_masses=n_masses,
                    device=device)
                
                #temp_recon_strain, _ = data_processing.normalise_data(temp_recon_strain, pre_model.norm_factor)

                m_recon_strain[i] = temp_recon_strain
                #m_recon_energy[i] = temp_recon_energy

            fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(9,7))
            for pi in range(3):
                for mi in range(2):
                    ax[mi,pi].boxplot(np.abs(multi_coeff_samples[:,mi,pi]), showfliers=False)
                    ax[mi,pi].plot(np.arange(len(source_coeffs[mi][pi])) + 1, np.abs(source_coeffs[mi][pi]), label="source", color="C0", ls="--")
     
            ax[2,0].hist(np.log(multi_mass_samples[:,0]), bins=100)
            ax[2,1].hist(np.log(multi_mass_samples[:,1]), bins=100)
            ax[2,0].axvline(np.log(source_masses[0]), color="r")
            ax[2,1].axvline(np.log(source_masses[1]), color="r")
            msun=1.0e30
            lmsun = np.log(msun)
            #ax[2,1].set_xlim([1e-6*msun, 7e-3*msun])
            #ax[2,0].set_xlim([1.8*msun, 2.2*msun])
            #ax[2,1].set_xlim([lmsun-4, lmsun])
            #ax[2,0].set_xlim([lmsun-1, lmsun+1])
            #ax[0,0].legend()
            fig.savefig(os.path.join(plot_out, f"test_power_{batch}.png"))

            fig, ax = plt.subplots( nrows=3, figsize=(9,7))
            for pi in range(3):
                ax[pi].boxplot(m_recon_strain[:,pi], showfliers=False)
                ax[pi].plot(np.arange(len(source_strain[pi])) + 1, source_strain[pi], label="source", color="C0", ls="--")
     
            #ax[2,1].set_xlim([1e-6*msun, 7e-3*msun])
            #ax[2,0].set_xlim([1.8*msun, 2.2*msun])
            #ax[2,1].set_xlim([lmsun-4, lmsun])
            #ax[2,0].set_xlim([lmsun-1, lmsun+1])
            #ax[0,0].legend()
            fig.savefig(os.path.join(plot_out, f"test_strain_{batch}.png"))
            #print("strainmax", np.max(source_strain), np.max(m_recon_strain))
            #print("labelmax", np.max(source_coeffs), np.max(multi_coeff_samples))
            if n_masses == 2:
                #print(np.shape(m_recon_masses))
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


            #############################
            ## SAVE THE DATA
            #############################

            with h5py.File(os.path.join(data_out, f"data_{batch}.hdf5"), "w") as f:
                f.create_dataset("recon_timeseries", data=m_recon_tseries)
                f.create_dataset("recon_strain", data=m_recon_strain)
                f.create_dataset("recon_masses", data=m_recon_masses)
                f.create_dataset("source_timeseries", data=source_tseries)
                f.create_dataset("source_strain", data=data[0].cpu().numpy())
                f.create_dataset("source_strain_signal_only", data=source_strain)
                f.create_dataset("source_masses", data=source_masses)
                f.create_dataset("source_basis", data=source_coeffs)
                f.create_dataset("recon_basis", data=multi_coeff_samples)

            if make_plots:
                """
                plotting.plot_sample_positions(
                    upsample_times, 
                    source_tseries, 
                    m_recon_tseries, 
                    n_dimensions, 
                    n_masses,
                    fname = os.path.join(plot_out, f"samples_positions_{batch}.png"))
                """
                plotting.plot_dimension_projection(
                    m_recon_tseries[:10], 
                    source_tseries, 
                    fname=os.path.join(plot_out, f"dim_projection_{batch}.png"), 
                    alpha=0.2)
                """
                plotting.plot_sample_separations(
                    upsample_times, 
                    source_tseries, 
                    m_recon_tseries, 
                    fname=os.path.join(plot_out,f"separations_{batch}.png"))
                """
                print("source_Strain", np.shape(source_strain))
                plotting.plot_sampled_reconstructions(
                    upsample_times, 
                    detectors, 
                    m_recon_strain, 
                    source_strain, 
                    fname = os.path.join(plot_out,f"recon_strain_dist_{batch}.png"))

                plotting.plot_mass_distributions(
                    m_recon_masses,
                    source_masses,
                    fname=os.path.join(plot_out,f"massdistributions_{batch}.png"))
                
                """
                print("line of sight ani")
                make_animations.line_of_sight_animation(
                    m_recon_tseries, 
                    m_recon_masses, 
                    source_tseries, 
                    source_masses, 
                    os.path.join(plot_out,f"2d_massdist_{batch}.gif"))
                """
                
                make_animations.heatmap_projections(
                    m_recon_tseries, 
                    m_recon_masses, 
                    source_tseries, 
                    source_masses, 
                    os.path.join(plot_out,f"heatmap_projections_{batch}.gif"),
                    duration=5)
                """
                make_animations.make_distribution_projections(
                    plot_out, 
                    batch, 
                    m_recon_tseries, 
                    m_recon_masses, 
                    source_tseries, 
                    source_masses,
                    strain=m_recon_strain,
                    true_strain=source_strain,
                    duration=8)
                """
                """
                print("3d dist")
                make_animations.make_3d_distribution(
                    plot_out, 
                    m_recon_tseries[:n_animate_samples], 
                    m_recon_masses[:n_animate_samples], 
                    source_tseries, 
                    source_masses,
                    fname = os.path.join(plot_out, f"multi_animation_{batch}.gif"))
                
                print("zproj")
                make_animations.make_3d_distribution_zproj(
                    plot_out, 
                    batch, 
                    m_recon_tseries, 
                    m_recon_masses, 
                    source_tseries, 
                    source_masses)
                """