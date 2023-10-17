


def test_model_3d(model, pre_model, dataloader, times, n_masses, chebyshev_order, n_dimensions, detectors, window, root_dir, device):
    """_summary_

    Args:
        model (_type_): _description_
        pre_model (_type_): _description_
        dataloader (_type_): _description_
        times (_type_): _description_
        n_masses (_type_): _description_
        chebyshev_order (_type_): _description_
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
            source_coeffs, source_masses, source_tseries = get_dynamics(label[0].cpu().numpy(), times, n_masses, chebyshev_order, n_dimensions)
            recon_coeffs, recon_masses, recon_tseries = get_dynamics(coeffmass_samples[0], times, n_masses, chebyshev_order, n_dimensions)

            fig, ax = plt.subplots(nrows = 4)
    
            recon_weighted_coeffs = np.sum(recon_coeffs * recon_masses[:, None, None], axis=0)
            source_weighted_coeffs = np.sum(source_coeffs * source_masses[:, None, None], axis=0)

            recon_strain_tensor = generate_3d_derivative(recon_weighted_coeffs, times)
            source_strain_tensor = generate_3d_derivative(source_weighted_coeffs, times)

            recon_strain = []
            source_strain = []
            for detector in detectors:
                recon_strain.append(compute_strain(recon_strain_tensor, detector=detector))
                source_strain.append(compute_strain(source_strain_tensor, detector=detector))

            for i in range(len(detectors)):
                print(np.shape(times), np.shape(recon_strain))

                ax[i].plot(times, recon_strain[i], label="recon")
                ax[i].plot(times, source_strain[i], label="source")
                ax[i].plot(times, data[0][i].cpu().numpy(), label="source data")

            fig.savefig(os.path.join(plot_out, f"reconstructed_{batch}.png"))

            make_3d_animation(plot_out, batch, recon_tseries, recon_masses, source_tseries, source_masses)


            nsamples = 50
            multi_coeffmass_samples = model(input_data).sample((nsamples, )).cpu().numpy()

            #print("multishape", multi_coeffmass_samples.shape)
            m_recon_tseries, m_recon_masses = np.zeros((nsamples, n_masses, n_dimensions, len(times))), np.zeros((nsamples, n_masses))
            for i in range(nsamples):
                #print(np.shape(multi_coeffmass_samples[i]))
                t_co, t_mass, t_time = get_dynamics(multi_coeffmass_samples[i][0], times, n_masses, chebyshev_order, n_dimensions)
                m_recon_tseries[i] = t_time
                m_recon_masses[i] = t_mass

            make_3d_distribution(plot_out, batch, m_recon_tseries, m_recon_masses, source_tseries, source_masses)


def load_models(root_dir):

    pre_model = nn.Sequential(
        nn.Conv1d(len(config["detectors"]), 32, 8, padding="same"),
        nn.ReLU(),
        nn.Conv1d(32, 32, 4),
        nn.ReLU(),
        nn.Conv1d(32, 16, 4),
        nn.ReLU(),
        nn.MaxPool1d(2),
        nn.Flatten(),
        nn.LazyLinear(n_context)
    ).to(config["device"])

    model = zuko.flows.spline.NSF(n_features, context=n_context, transforms=config["ntransforms"], bins=config["nsplines"], hidden_features=config["hidden_features"]).to(config["device"])
    
    weights = torch.load(os.path.join(config["root_dir"],"test_model.pt"))

    pre_model.load_state_dict(weights["pre_model_state_dict"])

    model.load_state_dict(weights["model_state_dict"])

    return pre_model, model

def chirp_model(times, f, M, ):

    G = 
    c = 3e8
    fdot = 96/5 * np.pi**(8/3) * (G*M/c**3)**(5/3) f**(11/3)
    return np.cos(2*np.pi*(f*t + fdot*t**2))

def run_chirp_test(config):

    pre_model, model = load_models(config["root_dir"])





if __name__ == "__main__":

    config = dict(
        n_data = 600000,
        batch_size = 512,
        chebyshev_order = 10,
        n_masses = 2,
        n_dimensions = 3,
        detectors=["H1", "L1", "V1"],
        sample_rate = 128,
        n_epochs = 500,
        window="hann",
        learning_rate = 2e-4,
        device = "cuda:0",
        nsplines = 6,
        ntransforms = 6,
        hidden_features = [256, 256, 256],
        root_dir = "test_model_3d_3det_antenna_hannwindow_1"
    )