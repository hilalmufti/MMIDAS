import torch
from torch import nn, optim
from torch.nn import functional as F

EPS = 1e-12


class VAE(nn.Module):
    def __init__(self, input_dim, lowD_dim, latent_spec, temperature=.67, use_cuda=True):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_spec : dict
            Specifies latent distribution. For example:
            {'cont': 10, 'disc': [10, 4, 3]} encodes 10 normal variables and
            3 gumbel softmax variables of dimension 10, 4 and 3. A latent spec
            can include both 'cont' and 'disc' or only 'cont' or only 'disc'.

        temperature : float
            Temperature for gumbel softmax distribution.

        use_cuda : bool
            If True moves model to GPU
        """
        super(VAE, self).__init__()
        self.use_cuda = use_cuda

        # Parameters
        self.is_continuous = 'cont' in latent_spec
        self.is_discrete = 'disc' in latent_spec
        self.latent_spec = latent_spec
        self.temperature = temperature
        self.lowD_dim = lowD_dim  # Hidden dimension of linear layer
        self.dp = nn.Dropout(.5) 

        # Calculate dimensions of latent distribution
        self.latent_cont_dim = 0
        self.latent_disc_dim = 0
        self.num_disc_latents = 0
        if self.is_continuous:
            self.latent_cont_dim = self.latent_spec['cont']
        if self.is_discrete:
            self.latent_disc_dim += sum([dim for dim in self.latent_spec['disc']])
            self.num_disc_latents = len(self.latent_spec['disc'])
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim

        # Define encoder layers
        # Intial layer
        encoder_layers = [
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=100,
                           eps=1e-10, momentum=0.1,
                           affine=False)
        ]

        # Add final layers
        encoder_layers += [
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=100,
                           eps=1e-10, momentum=0.1,
                           affine=False),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=100,
                           eps=1e-10, momentum=0.1,
                           affine=False),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=100,
                           eps=1e-10, momentum=0.1,
                           affine=False)
        ]

        # Define encoder
        self.img_to_features = nn.Sequential(*encoder_layers)

        # Map encoded features into a hidden vector which will be used to
        # encode parameters of the latent distribution
        self.features_to_hidden = nn.Sequential(
            nn.Linear(100, self.lowD_dim),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.lowD_dim,
                           eps=1e-10, momentum=0.1,
                           affine=False)
        )

        # Encode parameters of latent distribution
        if self.is_continuous:
            self.fc_mean = nn.Linear(self.lowD_dim, self.latent_cont_dim)
            self.fc_log_var = nn.Linear(self.lowD_dim, self.latent_cont_dim)
        if self.is_discrete:
            # Linear layer for each of the categorical distributions
            fc_alphas = []
            for disc_dim in self.latent_spec['disc']:
                fc_alphas.append(nn.Linear(self.lowD_dim, disc_dim))
            self.fc_alphas = nn.ModuleList(fc_alphas)

        # Map latent samples to features to be used by generative model
        self.latent_to_features = nn.Sequential(
            nn.Linear(self.latent_dim, self.lowD_dim),
            nn.ReLU(),
            nn.Linear(self.lowD_dim, 100),
            nn.ReLU()
        )

        # Define decoder
        decoder_layers = []

        decoder_layers += [
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, input_dim),
            nn.ReLU()
        ]

        # Define decoder
        self.features_to_img = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """
        Encodes an image into parameters of a latent distribution defined in
        self.latent_spec.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data, shape (N, C, H, W)
        """
        batch_size = x.size()[0]

        # Encode image to hidden features
        features = self.img_to_features(self.dp(x))
        hidden = self.features_to_hidden(features)

        # Output parameters of latent distribution from hidden representation
        latent_dist = {}

        if self.is_continuous:
            latent_dist['cont'] = [self.fc_mean(hidden), self.fc_log_var(hidden)]

        if self.is_discrete:
            latent_dist['disc'] = []
            for fc_alpha in self.fc_alphas:
                latent_dist['disc'].append(F.softmax(fc_alpha(hidden), dim=1))

        return hidden, latent_dist

    def reparameterize(self, latent_dist):
        """
        Samples from latent distribution using the reparameterization trick.

        Parameters
        ----------
        latent_dist : dict
            Dict with keys 'cont' or 'disc' or both, containing the parameters
            of the latent distributions as torch.Tensor instances.
        """
        latent_sample = []

        if self.is_continuous:
            mean, logvar = latent_dist['cont']
            cont_sample = self.sample_normal(mean, logvar)
            latent_sample.append(cont_sample)

        if self.is_discrete:
            for alpha in latent_dist['disc']:
                disc_sample = self.sample_gumbel_softmax(alpha)
                latent_sample.append(disc_sample)

        # Concatenate continuous and discrete samples into one large sample
        return torch.cat(latent_sample, dim=1)

    def sample_normal(self, mean, logvar):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.zeros(std.size()).normal_()
            if self.use_cuda:
                eps = eps.cuda()
            return mean + std * eps
        else:
            # Reconstruction mode
            return mean

    def sample_gumbel_softmax(self, alpha):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        """
        if self.training:
            # Sample from gumbel distribution
            unif = torch.rand(alpha.size())
            if self.use_cuda:
                unif = unif.cuda()
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            # Reparameterize to create gumbel softmax sample
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1)
        else:
            # # In reconstruction mode, pick most likely sample
            # _, max_alpha = torch.max(alpha, dim=1)
            # one_hot_samples = torch.zeros(alpha.size())
            # # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # # max_alpha. Note the view is because scatter_ only accepts 2D
            # # tensors.
            # one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
            # if self.use_cuda:
            #     one_hot_samples = one_hot_samples.cuda()
            # return one_hot_samples
            unif = torch.rand(alpha.size())
            if self.use_cuda:
                unif = unif.cuda()

            # Reparameterize to create gumbel softmax sample
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha) / self.temperature
            return F.softmax(logit, dim=1)


    def decode(self, latent_sample):
        """
        Decodes sample from latent distribution into an image.

        Parameters
        ----------
        latent_sample : torch.Tensor
            Sample from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        features = self.latent_to_features(latent_sample)
        return self.features_to_img(features)

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (N, C, H, W)
        """
        hidden, latent_dist = self.encode(x)
        latent_sample = self.reparameterize(latent_dist)
        recon_x = self.decode(latent_sample)
        return recon_x, hidden, latent_dist