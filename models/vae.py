import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activations, device="cpu"):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.device = device

        self.linear_nets = nn.Sequential()
        prev_dim = input_dim
        for i, (hidden_dim, activation) in enumerate(zip(hidden_dims, activations)):
            self.linear_nets.add_module("fc_{}".format(i), nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
            if activation == "relu":
                self.linear_nets.add_module("act_{}".format(i), nn.ReLU())
            elif activation == "lrelu":
                self.linear_nets.add_module("act_{}".format(i), nn.LeakyReLU(0.2))
            elif activation == "sigmoid":
                self.linear_nets.add_module("act_{}".format(i), nn.Sigmoid())
            elif activation == "softmax":
                self.linear_nets.add_module("act_{}".format(i), nn.Softmax(dim=1))
            elif activation == "tanh":
                self.linear_nets.add_module("act_{}".format(i), nn.Tanh())

        self.to(self.device)

    def forward(self, x):
        return self.linear_nets(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        h = self.LeakyReLU(self.FC_input(x))
        h = self.LeakyReLU(self.FC_input2(h))
        mean = self.FC_mean(h)
        log_var = self.FC_var(h)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))

        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, n_layers=3, activation="lrelu", out_activation=None,
                 device="cpu"):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        # encoder
        self.mean_z = MLP(input_dim=input_dim,
                          hidden_dims=[hidden_dim] * (n_layers - 1) + [latent_dim],
                          activations=[activation] * (n_layers - 1) + [out_activation],
                          device=device)
        self.log_var_z = MLP(input_dim=input_dim,
                             hidden_dims=[hidden_dim] * (n_layers - 1) + [latent_dim],
                             activations=[activation] * (n_layers - 1) + [out_activation],
                             device=device)

        # decoder
        self.decoder = MLP(input_dim=latent_dim,
                           hidden_dims=[hidden_dim] * (n_layers - 1) + [input_dim],
                           activations=[activation] * (n_layers - 1) + [out_activation],
                           device=device)


    def encode(self, x):
        mean = self.mean_z(x)
        log_var = self.log_var_z(x)
        return mean, log_var

    def decode(self, x):
        return self.decoder(x)

    def reparameterization(self, mean, std):
        eps = torch.randn_like(std).to(self.device)
        z = mean + std * eps
        return z

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat, mean, log_var

    def reconstruct(self, x, sample=False):
        mean, log_var = self.encode(x)
        z = mean
        if sample:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat


class iVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, auxiliary_dim, hidden_dim, n_layers=3, activation="lrelu",
                 out_activation=None, device="cpu", prior_mean=False):
        super(iVAE, self).__init__()
        self.latent_dim = latent_dim

        self.device = device

        # prior params
        self.prior_mean = prior_mean
        if self.prior_mean:
            self.prior_mean_z = MLP(input_dim=auxiliary_dim,
                                    hidden_dims=[hidden_dim] * (n_layers - 1) + [latent_dim],
                                    activations=[activation] * (n_layers - 1) + [out_activation],
                                    device=device)
        self.prior_log_var_z = MLP(input_dim=auxiliary_dim,
                                   hidden_dims=[hidden_dim] * (n_layers - 1) + [latent_dim],
                                   activations=[activation] * (n_layers - 1) + [out_activation],
                                   device=device)

        # encoder params
        self.mean_z = MLP(input_dim=input_dim + auxiliary_dim,
                          hidden_dims=[hidden_dim] * (n_layers - 1) + [latent_dim],
                          activations=[activation] * (n_layers - 1) + [out_activation],
                          device=device)
        self.log_var_z = MLP(input_dim=input_dim + auxiliary_dim,
                             hidden_dims=[hidden_dim] * (n_layers - 1) + [latent_dim],
                             activations=[activation] * (n_layers - 1) + [out_activation],
                             device=device)

        # decoder params
        self.decoder = MLP(input_dim=latent_dim,
                           hidden_dims=[hidden_dim] * (n_layers - 1) + [input_dim],
                           activations=[activation] * (n_layers - 1) + [out_activation],
                           device=device)

    def encode(self, x, w):
        xw = torch.cat((x, w), 1)
        mean = self.mean_z(xw)
        log_var = self.log_var_z(xw)
        return mean, log_var

    def decode(self, x):
        return self.decoder(x)

    def prior(self, w):
        log_var_z = self.prior_log_var_z(w)
        if self.prior_mean:
            mean_z = self.prior_mean_z(w)
        else:
            mean_z = torch.zeros_like(log_var_z).to(self.device)
        return mean_z, log_var_z

    def reparameterization(self, mean, std):
        eps = torch.randn_like(std).to(self.device)
        z = mean + std * eps
        return z

    def forward(self, x, w):
        prior_log_var = self.prior_log_var_z(w)

        if self.prior_mean:
            prior_mean = self.prior_mean_z(w)
        else:
            prior_mean = torch.zeros_like(prior_log_var).to(self.device)
        mean, log_var = self.encode(x, w)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat, mean, log_var, prior_mean, prior_log_var

    def reconstruct(self, x, w, sample=False):
        mean, log_var = self.encode(x, w)
        z = mean
        if sample:
            z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decode(z)
        return x_hat
