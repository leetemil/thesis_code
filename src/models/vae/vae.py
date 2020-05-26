# Inspired by:
#   https://github.com/pytorch/examples/blob/master/vae/main.py
#   https://vxlabs.com/2017/12/08/variational-autoencoder-in-pytorch-commented-and-annotated/
from enum import Enum

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

from .bayesian import bayesian, Bayesian
from ..utils import smooth_one_hot

class VAE(nn.Module):
    """Variational Auto-Encoder for protein sequences with optional variational approximation of global parameters"""

    def __init__(self, layer_sizes, num_tokens, z_samples = 4, dropout = 0.5, use_bayesian = True, num_patterns = 4, inner_CW_dim = 40, use_param_loss = True, use_dictionary = False, label_smoothing = 0.0, warm_up = 0, weight_loss = True):
        super().__init__()

        assert len(layer_sizes) >= 2

        self.layer_sizes = layer_sizes
        self.num_tokens = num_tokens
        self.sequence_length = self.layer_sizes[0] // self.num_tokens
        self.z_samples = z_samples
        self.dropout = dropout
        self.bayesian = use_bayesian
        self.num_patterns = num_patterns
        self.inner_CW_dim = inner_CW_dim
        self.use_param_loss = use_param_loss
        self.use_dictionary = use_dictionary
        self.label_smoothing = label_smoothing
        self.warm_up = warm_up
        self.warm_up_scale = 0
        self.weight_loss = weight_loss

        bottleneck_idx = layer_sizes.index(min(layer_sizes))

        # Construct encode layers except last ones
        encode_layers = []
        layer_sizes_doubles = [(s1, s2) for s1, s2 in zip(layer_sizes[:bottleneck_idx], layer_sizes[1:])]
        for s1, s2 in layer_sizes_doubles[:-1]:
            layer = nn.Linear(s1, s2)
            # Initialization taken from Deep Sequence paper
            init.xavier_normal_(layer.weight)
            init.constant_(layer.bias, 0.1)
            encode_layers.append(layer)
            encode_layers.append(nn.ReLU(inplace = True))
            encode_layers.append(nn.Dropout(self.dropout))
        self.encode_layers = nn.Sequential(*encode_layers)

        # Last two layers to get to bottleneck size
        s1, s2 = layer_sizes_doubles[-1]
        self.encode_mean = nn.Linear(s1, s2)
        self.encode_logvar = nn.Linear(s1, s2)

        # Initialize last encode layers
        init.xavier_normal_(self.encode_mean.weight)
        init.constant_(self.encode_mean.bias, 0.1)

        init.xavier_normal_(self.encode_logvar.weight)
        init.constant_(self.encode_logvar.bias, -10)

        # Construct decode layers
        if self.bayesian:
            decode_mod = bayesian
        else:
            def decode_mod(x, *args, **kwargs):
                return x

        decode_layers = []
        layer_sizes_doubles = [(s1, s2) for s1, s2 in zip(layer_sizes[bottleneck_idx:], layer_sizes[bottleneck_idx + 1:])]
        for s1, s2 in layer_sizes_doubles[:-2]:
            decode_layers.append(decode_mod(nn.Linear(s1, s2)))
            decode_layers.append(nn.ReLU(inplace = True))
            decode_layers.append(nn.Dropout(self.dropout))

        # Second-to-last decode layer has sigmoid activation
        s1, s2 = layer_sizes_doubles[-2]
        decode_layers.append(decode_mod(nn.Linear(s1, s2)))
        decode_layers.append(nn.ReLU()) # Experimental
        decode_layers.append(nn.Dropout(self.dropout))

        if not self.use_dictionary:
            # Last decode layer has no activation
            s1, s2 = layer_sizes_doubles[-1]
            decode_layers.append(decode_mod(nn.Linear(s1, s2)))
        else:
            if self.bayesian:
                self.b3_mean = nn.Parameter(torch.Tensor([0.1] * self.num_tokens * self.sequence_length))
                self.b3_logvar = nn.Parameter(torch.Tensor([-10.0] * self.num_tokens * self.sequence_length))
                self.l_mean = nn.Parameter(torch.Tensor([1]))
                self.l_logvar = nn.Parameter(torch.Tensor([-10.0]))
            else:
                # todo: initialize randomly
                self.b3 = nn.Parameter(torch.Tensor([1]))
                self.l = nn.Parameter(torch.Tensor([0.1] * self.num_tokens * self.sequence_length))

            self.W3 = decode_mod(nn.Linear(s2, self.inner_CW_dim * self.sequence_length, bias = False), "weight")
            self.S = decode_mod(nn.Linear(self.sequence_length, s2 // self.num_patterns, bias = False), "weight")
            self.C = decode_mod(nn.Linear(self.num_tokens, self.inner_CW_dim, bias = False), "weight")

        self.decode_layers = nn.Sequential(*decode_layers)

    def encode(self, x):
        x = F.one_hot(x, self.num_tokens).to(torch.float).flatten(1)

        # Encode x by sending it through all encode layers
        x = self.encode_layers(x)
        mean = self.encode_mean(x)
        logvar = self.encode_logvar(x)

        return Normal(mean, logvar.mul(0.5).exp())

    def decode(self, z):
        # Send z through all decode layers
        z = self.decode_layers(z)

        if not self.use_dictionary:
            z = z.view(z.size(0), self.sequence_length, self.num_tokens)
            z = torch.log_softmax(z, dim = -1)
            return z

        if self.bayesian:
            self.S.sample_new_weight()
            self.W3.sample_new_weight()
            self.C.sample_new_weight()
            l = Normal(self.l_mean, self.l_logvar.mul(0.5).exp()).rsample()
            b3 = Normal(self.b3_mean, self.b3_logvar.mul(0.5).exp()).rsample()
        else:
            l = self.l
            b3 = self.b3

        S = torch.sigmoid(self.S.weight.repeat(self.num_patterns, 1))

        W3 = self.W3.weight.view(self.layer_sizes[-2] * self.sequence_length, -1)

        W_out = W3 @ self.C.weight

        W_out = W_out.view(-1, self.sequence_length, self.num_tokens)
        W_out = W_out * S.unsqueeze(2)
        W_out = W_out.view(-1, self.sequence_length * self.num_tokens)

        h3 = F.linear(z, W_out.T, b3)

        h3 = h3 * torch.log(1 + l.exp())

        h3 = h3.view(h3.size(0), self.sequence_length, self.num_tokens)
        h3 = torch.log_softmax(h3, dim = -1)

        return h3

    def sample(self, z):
        z = self.decode(z)
        sample = z.exp().argmax(dim = -1)
        return sample

    def get_representation(self, xb):
        return self.encode(xb).mean

    def sample_random(self, batch_size = 1):
        z = torch.randn(batch_size, self.layer_sizes[-1])
        return self.sample(z)

    def reconstruct(self, x):
        encoded_distribution = self.encode(x)
        return self.sample(encoded_distribution.mean)

    def forward(self, x, weights, neff):
        if self.training:
            self.warm_up_scale += 1 / (1 + self.warm_up)
            self.warm_up_scale = min(1, self.warm_up_scale)
            warm_up_scale = self.warm_up_scale
        else:
            warm_up_scale = 1
            self.weight_loss = True

        # Forward pass + loss + metrics
        encoded_distribution = self.encode(x)
        z = encoded_distribution.rsample((self.z_samples,))
        recon_x = self.decode(z.flatten(0, 1))
        total_loss, nll_loss, kld_loss, param_kld = self.vae_loss(recon_x, x, encoded_distribution, weights, neff, warm_up_scale)

        # Metrics
        metrics_dict = {}

        # Accuracy
        with torch.no_grad():
            acc = (self.decode(encoded_distribution.mean).exp().argmax(dim = -1) == x).to(torch.float).mean().item()
            metrics_dict["accuracy"] = acc
            metrics_dict["nll_loss"] = nll_loss
            metrics_dict["kld_loss"] = kld_loss
            metrics_dict["param_kld"] = param_kld

        return total_loss, metrics_dict

    def summary(self):
        num_params = sum(p.numel() for p in self.parameters())

        return (f"Variational Auto-Encoder summary:\n"
                f"  Layer sizes: {self.layer_sizes}\n"
                f"  Parameters: {num_params:,}\n"
                f"  Bayesian: {self.bayesian}\n")

    def get_predictions(self, xb):
        """
        Returns log-softmax distributions of amino acids over the input sequences.

        Returns:
        Tensor: shape (batch size, num tokens, seq length)
        """
        encoded_distribution = self.encode(xb)
        z = encoded_distribution.rsample((self.z_samples,))
        recon_xb = self.decode(z.flatten(0, 1))
        return recon_xb.permute(0, 2, 1)


    def protein_logp(self, x):
        encoded_distribution = self.encode(x)
        kld = self.kld_loss(encoded_distribution)
        # z = encoded_distribution.mean
        z = encoded_distribution.rsample()
        recon_x = self.decode(z).permute(0, 2, 1)
        logp = F.nll_loss(recon_x, x, reduction = "none").mul(-1).sum(1)
        elbo = logp + kld

        # amino acid probabilities are independent conditioned on z
        return elbo, logp, kld

    def recon_loss(self, recon_x, x):
        # How well do input x and output recon_x agree?
        recon_x = recon_x.view(self.z_samples, -1, recon_x.size(1), self.num_tokens).permute(1, 2, 0, 3)
        x = x.unsqueeze(-1).expand(-1, -1, self.z_samples)

        smooth_target = smooth_one_hot(x, self.num_tokens, self.label_smoothing)
        loss = -(smooth_target * recon_x).sum(-1)
        loss = loss.mean(-1).sum(-1)
        return loss

        # How well do input x and output recon_x agree?
        # recon_x = recon_x.view(self.z_samples, -1, recon_x.size(1), self.num_tokens).permute(1, 3, 2, 0)
        # x = x.unsqueeze(-1).expand(-1, -1, self.z_samples)

        # nll = F.nll_loss(recon_x, x, reduction = "none").mean(-1).sum(1)

        # # amino acid probabilities are independent conditioned on z
        # return nll

    def kld_loss(self, encoded_distribution):
        prior = Normal(torch.zeros_like(encoded_distribution.mean), torch.ones_like(encoded_distribution.variance))
        kld = kl_divergence(encoded_distribution, prior).sum(dim = 1)

        return kld

    def global_parameter_kld(self):

        global_kld = 0

        for layer in self.decode_layers:
            if isinstance(layer, torch.nn.Linear):
                # get weight and bias distributions
                weight_mean = layer.weight_mean
                weight_std = layer.weight_logvar.mul(1/2).exp()
                bias_mean = layer.bias_mean
                bias_std = layer.bias_logvar.mul(1/2).exp()

                q_weight = Normal(weight_mean, weight_std)
                q_bias = Normal(bias_mean, bias_std)

                # all layers has a unit Gaussian prior
                p_weight = Normal(torch.zeros_like(weight_mean), torch.ones_like(weight_std))
                p_bias = Normal(torch.zeros_like(bias_mean), torch.ones_like(bias_std))

                weight_kld = kl_divergence(q_weight, p_weight).sum()
                bias_kld = kl_divergence(q_bias, p_bias).sum()
                global_kld += weight_kld + bias_kld

        if not self.use_dictionary:
            return global_kld

        # W3 loss
        W3_distribution = Normal(self.W3.weight_mean, self.W3.weight_logvar.mul(0.5).exp())
        W3_unit_guassian = Normal(torch.zeros_like(self.W3.weight_mean), torch.ones_like(self.W3.weight_logvar))
        global_kld += kl_divergence(W3_distribution, W3_unit_guassian).sum()

        b3_distribution = Normal(self.b3_mean, self.b3_logvar.mul(0.5).exp())
        b3_unit_guassian = Normal(torch.zeros_like(self.b3_mean), torch.ones_like(self.b3_logvar))
        global_kld += kl_divergence(b3_distribution, b3_unit_guassian).sum()

        C_distribution = Normal(self.C.weight_mean, self.C.weight_logvar.mul(0.5).exp())
        C_unit_guassian = Normal(torch.zeros_like(self.C.weight_mean), torch.ones_like(self.C.weight_logvar))
        global_kld += kl_divergence(C_distribution, C_unit_guassian).sum()

        S_distribution = Normal(self.S.weight_mean, self.S.weight_logvar.mul(0.5).exp())
        S_unit_guassian = Normal(torch.zeros_like(self.S.weight_mean) - 12.36, torch.exp(torch.zeros_like(self.S.weight_logvar) + 0.602))
        global_kld += kl_divergence(S_distribution, S_unit_guassian).sum()

        l_distribution = Normal(self.l_mean, self.l_logvar.mul(0.5).exp())
        l_unit_guassian = Normal(0, 1)
        global_kld += kl_divergence(l_distribution, l_unit_guassian).sum()

        return global_kld

    def vae_loss(self, recon_x, x, encoded_distribution, weights, neff, warm_up_scale):
        recon_loss = self.recon_loss(recon_x, x)
        kld_loss = self.kld_loss(encoded_distribution)

        if self.weight_loss:
            recon_loss *= weights
            kld_loss *= weights
            # Emil: This change is not tested. Maybe revert to simply always do .mean()
            recon_kld_loss = (recon_loss + kld_loss).sum()

        else:
            recon_kld_loss = torch.mean(recon_loss + kld_loss)

        if self.bayesian and self.use_param_loss:
            param_kld = self.warm_up_scale * self.global_parameter_kld() / neff
            total_loss = recon_kld_loss + param_kld
        else:
            param_kld = torch.zeros(1) + 1e-5
            total_loss = recon_kld_loss

        return total_loss, recon_loss.mean().item(), kld_loss.mean().item(), param_kld.item()

    def save(self, f):
        args_dict = {
            "layer_sizes": self.layer_sizes,
            "num_tokens": self.num_tokens,
            "z_samples": self.z_samples,
            "dropout": self.dropout,
            "use_bayesian": self.bayesian,
            "num_patterns": self.num_patterns,
            "inner_CW_dim": self.inner_CW_dim,
            "use_param_loss": self.use_param_loss,
            "use_dictionary": self.use_dictionary,
            "label_smoothing": self.label_smoothing,
            "warm_up": self.warm_up,
        }

        torch.save({
            "name": "VAE",
            "state_dict": self.state_dict(),
            "args_dict": args_dict,
        }, f)
