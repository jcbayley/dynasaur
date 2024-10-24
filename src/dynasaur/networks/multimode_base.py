from glasflow.nflows.distributions.base import Distribution
from glasflow.nflows.distributions import StandardNormal
from torch.distributions import MultivariateNormal
import torch
from torch.distributions import Normal
import numpy as np

class CompositeLatentDistribution(Distribution):
    def __init__(self, n_masses, n_dimensions):
        super().__init__()
        self.n_masses = n_masses
        self.n_dimensions = n_dimensions
        self.latent_position = MADEMoG(features=n_masses*n_dimensions, num_mixture_components=2, hidden_features=64)
        self.latent_masses = StandardNormal(shape=n_masses)  # Mixture for (x1, x2)

    def log_prob(self, inputs):
        # Split the inputs into (x1, x2) and m
        position, m = inputs[:, :-self.n_masses], inputs[:, -self.n_masses:]
        
        # Compute log probability for each part
        log_prob_x1_x2 = self.latent_position.log_prob(position)
        log_prob_m = self.latent_masses.log_prob(m)
        
        # Return the sum of log probabilities
        return log_prob_x1_x2 + log_prob_m

    def sample(self, num_samples):
        # Sample from the Mixture of Gaussians for (x1, x2)
        samples_x1_x2 = self.latent_position.sample(num_samples)
        
        # Sample from the Gaussian for m
        samples_m = self.latent_masses.sample(num_samples)
        
        # Concatenate the samples to form the full latent variable
        return torch.cat([samples_x1_x2, samples_m], dim=-1)


class SymmetricLatentDistribution(Distribution):
    def __init__(self, num_symmetric, num_unimodal):
        """
        Parameters:
        - num_symmetric: Number of symmetric variables (e.g., positions).
        - num_unimodal: Number of unimodal variables (e.g., masses).
        """
        super().__init__()
        
        self.num_symmetric = num_symmetric
        self.num_unimodal = num_unimodal

        # Define the two symmetric modes centered at +/-1 for each symmetric variable
        self.symmetric_means = torch.tensor([1.0, -1.0])  # Means at 1 and -1
        self.symmetric_std = torch.tensor(0.4)  # Unit standard deviation
        self.sym_dist = []
        for mean in self.symmetric_means:
            self.sym_dist.append(Normal(mean, self.symmetric_std))

        # Single Gaussian for unimodal variables (centered at 0 with unit variance)
        self.unimodal_mean = torch.tensor(0.0)
        self.unimodal_std = torch.tensor(1.0)
        self.unimodal_dist = Normal(self.unimodal_mean, self.unimodal_std)

    def log_prob(self, inputs):
        """Compute the log-probability for symmetric and unimodal parts."""
        # Split inputs into symmetric part (e.g., x1, x2) and unimodal part (e.g., m)
        symmetric_inputs, unimodal_inputs = inputs[:, :self.num_symmetric], inputs[:, self.num_symmetric:]

        # Log probability for symmetric part (mixture of two modes)
        log_probs_symmetric = [dist.log_prob(symmetric_inputs) for dist in self.sym_dist]
        
        # Stack log-probs and compute log-sum-exp for numerical stability
        log_probs_symmetric = torch.stack(log_probs_symmetric, dim=-1)
        log_prob_symmetric = torch.logsumexp(log_probs_symmetric, dim=-1) - torch.log(torch.tensor(2.0))  # Equal weighting of modes

        # Log probability for unimodal part
        log_prob_unimodal = self.unimodal_dist.log_prob(unimodal_inputs).sum(dim=-1)

        # Return combined log-probability
        return log_prob_symmetric.sum(dim=-1) + log_prob_unimodal

    def sample(self, num_samples):
        """Sample from the distribution (symmetric bimodal + unimodal)."""
        # Sample from each mode for the symmetric part (randomly pick +1 or -1)
        component_indices = torch.randint(0, 2, (num_samples, self.num_symmetric))
        samples_symmetric = self.symmetric_means[component_indices] + self.symmetric_std * torch.randn(num_samples, self.num_symmetric)

        # Sample from the unimodal Gaussian for the remaining variables
        samples_unimodal = self.unimodal_dist.sample((num_samples, self.num_unimodal))

        # Concatenate the symmetric and unimodal samples
        return torch.cat([samples_symmetric, samples_unimodal], dim=-1)