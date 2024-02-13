import torch.nn as nn
import torch
import torch.nn.functional as F


class myVAE(nn.Module):
    def __init__(self, latent_dim, input_dim=5000, n_dim=500):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, input_dim // 5)
        self.batch_fc1 = nn.BatchNorm1d(num_features=self.fc1.out_features)
                                        # eps=1e-10, momentum=0.1, affine=False)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features)
        self.batch_fc2 = nn.BatchNorm1d(num_features=self.fc2.out_features)
                                        # eps=1e-10, momentum=0.1, affine=False)
        self.fc3 = nn.Linear(self.fc2.out_features, n_dim)
        self.batch_fc3 = nn.BatchNorm1d(num_features=self.fc3.out_features)
                                        # eps=1e-10, momentum=0.1, affine=False)
        self.fc4 = nn.Linear(n_dim, n_dim)
        self.batch_fc4 = nn.BatchNorm1d(num_features=self.fc4.out_features)
                                        # eps=1e-10, momentum=0.1, affine=False)
        self.fc5 = nn.Linear(n_dim, n_dim // 5)
        self.batch_fc5 = nn.BatchNorm1d(num_features=self.fc5.out_features)
                                        # eps=1e-10, momentum=0.1, affine=False)
        self.fcs = nn.Linear(self.fc5.out_features, latent_dim)
        self.batch_fcs = nn.BatchNorm1d(num_features=self.fcs.out_features)
                                        # eps=1e-10, momentum=0.1, affine=False)
        self.fc6 = nn.Linear(self.fcs.out_features, n_dim // 5)
        self.batch_fc6 = nn.BatchNorm1d(num_features=self.fc6.out_features)
                                        # eps=1e-10, momentum=0.1, affine=False)
        self.fc7 = nn.Linear(self.fc6.out_features, n_dim)
        self.batch_fc7 = nn.BatchNorm1d(num_features=self.fc7.out_features)
                                        # eps=1e-10, momentum=0.1, affine=False)
        self.fc8 = nn.Linear(n_dim, n_dim)
        self.batch_fc8 = nn.BatchNorm1d(num_features=self.fc8.out_features)
                                        # eps=1e-10, momentum=0.1, affine=False)
        self.fc9 = nn.Linear(n_dim, input_dim // 5)
        self.batch_fc9 = nn.BatchNorm1d(num_features=self.fc9.out_features)
                                        # eps=1e-10, momentum=0.1, affine=False)
        self.fc10 = nn.Linear(self.fc9.out_features, self.fc9.out_features)
        self.batch_fc10 = nn.BatchNorm1d(num_features=self.fc10.out_features)
                                        # eps=1e-10, momentum=0.1, affine=False)
        self.fc11 = nn.Linear(self.fc10.out_features, input_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.batch_fc5(self.fc5(x)))
        s = self.batch_fcs(self.fcs(x))
        x = F.relu(self.batch_fc6(self.fc6(s)))
        x = F.relu(self.batch_fc7(self.fc7(x)))
        x = F.relu(self.batch_fc8(self.fc8(x)))
        x = F.relu(self.batch_fc9(self.fc9(x)))
        x = F.relu(self.batch_fc10(self.fc10(x)))

        return s, F.relu(self.fc11(x))

class VAE(nn.Module):
    def __init__(self, input_dim, fc_dim, latent_dim, p_drop, varitioanl):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.fc_dim = fc_dim
        self.latent_dim = latent_dim
        self.dp = nn.Dropout(p_drop)
        self.variational = varitioanl

        self.fc1 = nn.Linear(input_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc3 = nn.Linear(fc_dim, fc_dim)
        self.fc4 = nn.Linear(fc_dim, fc_dim)
        self.fc_mu = nn.Linear(fc_dim, latent_dim)
        self.fc_sigma= nn.Linear(fc_dim, latent_dim)
        self.batchNorm1d = torch.nn.BatchNorm1d(latent_dim, eps=1e-10, momentum=0.01)

        self.fc6 = nn.Linear(latent_dim, fc_dim)
        self.fc7 = nn.Linear(fc_dim, fc_dim)
        self.fc8 = nn.Linear(fc_dim, fc_dim)
        self.fc9 = nn.Linear(fc_dim, fc_dim)
        self.fc10 = nn.Linear(fc_dim, input_dim)


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(self.dp(x)))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        h4 = self.relu(self.fc4(h3))
        mu = self.batchNorm1d(self.fc_mu((h4)))
        if self.variational:
            var = F.sigmoid(self.fc_sigma(h4))
            return mu, var
        else:
            return mu

    def reparameterize(self, mu, var):
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h6 = self.relu(self.fc6(z))
        h7 = self.relu(self.fc7(h6))
        h8 = self.relu(self.fc8(h7))
        h9 = self.relu(self.fc9(h8))
        recon = self.relu(self.fc10(h9))
        return recon

    def forward(self, x):
        if self.variational:
            mu, var = self.encode(x)
            z = self.reparameterize(mu, var)
        else:
            z = self.encode(x)

        return self.decode(z), z

    def loss(self, x_recon, x):
        return F.mse_loss(x_recon, x, reduction='mean')
