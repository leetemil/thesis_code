import torch
from torch import nn
from torch.nn import functional as F

from .norm_conv import NormConv
from ..vae.bayesian import bayesian, Bayesian
from ..utils import layer_kld
from data import IUPAC_SEQ2IDX

class WaveNet(nn.Module):

    def __init__(self, input_channels, residual_channels, out_channels, stacks, layers_per_stack, total_samples, l2_lambda = 0, bias = True, dropout = 0.5, use_bayesian = False, backwards = False, multi_gpu = False):
        super().__init__()

        self.input_channels = input_channels
        self.residual_channels = residual_channels
        self.out_channels = out_channels
        self.stacks = stacks
        self.layers_per_stack = layers_per_stack
        self.total_samples = total_samples
        self.l2_lambda = l2_lambda
        self.bias = bias
        self.dropout = dropout
        self.bayesian = use_bayesian
        self.backwards = backwards
        self.activation = F.elu
        self.first_conv = NormConv(self.input_channels, self.residual_channels, self.bayesian, kernel_size = 1, bias = self.bias)
        self.last_conv_layer = NormConv(self.residual_channels, self.out_channels, self.bayesian, kernel_size = 1, bias = self.bias)

        self.multi_gpu = multi_gpu

        dilations = []
        for i in range(self.layers_per_stack):
            dilations.append(2**i)

        blocks = []
        for _ in range(self.stacks):
            blocks.append(
                WaveNetStack(
                    dilations = dilations,
                    channels = self.residual_channels,
                    dropout = self.dropout,
                    kernel_size = 2,
                    activation = self.activation,
                    bayesian = self.bayesian,
                )
            )

        self.dilated_conv_stack = nn.Sequential(*blocks)

    def preprocess(self, xb):
        if self.backwards:
            lengths = (xb != 0).sum(dim = 1)
            for seq, length in zip(xb, lengths):
                seq[1:length - 1] = reversed(seq[1:length - 1])
        # one-hot encode and permute to (batch size x channels x length)
        xb_encoded = F.one_hot(xb, self.input_channels).to(torch.float).permute(0, 2, 1)
        return xb_encoded

    def get_predictions(self, xb):
        """
        Returns log-softmax distributions of amino acids over the input sequences.

        Returns:
        Tensor: shape (batch size, num tokens, seq length)
        """
        # encode and, if needed, reverse sequences
        xb = self.preprocess(xb)

        pred = self.first_conv(xb)
        pred = self.dilated_conv_stack(pred)
        pred = self.last_conv_layer(pred)

        return F.log_softmax(pred, dim = 1)

    def protein_logp(self, xb):
        loss, _ = self(xb, loss_reduction = "none")
        log_probabilities = -1 * loss.sum(dim = 1)
        return log_probabilities

    def parameter_kld(self):
        kld = layer_kld(self.first_conv.layer) if isinstance(self.first_conv, NormConv) else 0
        kld += layer_kld(self.last_conv_layer.layer) if isinstance(self.last_conv_layer, NormConv) else 0

        # get loss from stack layers
        for stack in self.dilated_conv_stack:
            kld += stack.parameter_kld()

        return kld

    def get_representation(self, xb, input_mask, variant = "mean"):
        # encode and, if needed, reverse sequences
        xb = self.preprocess(xb)
        xb = self.first_conv(xb)
        xb = self.dilated_conv_stack(xb)

        if variant == "mean":
            # return mean of channels across each sequence. Representation is shape num_channels
            representation = xb.mul(input_mask.unsqueeze(1)).sum(2).div(input_mask.sum(1).unsqueeze(1))

        elif variant == "last":
            raise NotImplementedError("Last not implemented")
            representation = xb[:, :, -1]

        else:
            raise ValueError(f"Representation variant {variant} is not supported.")

        return xb.permute(0, 2, 1), representation

    def forward(self, xb, weights = None, neff = None, loss_reduction = "mean"):
        pred = self.get_predictions(xb)

        # Calculate loss
        mask = xb >= IUPAC_SEQ2IDX["A"]
        true = (xb * mask)[:, 1:-1]
        pred = pred[:, :, :-2]

        # Compare each timestep in cross entropy loss
        if loss_reduction == "mean":
            nll_loss = F.nll_loss(pred, true, ignore_index = 0, reduction = "none").sum(1)
            lengths = mask.sum(1)
            nll_loss /= lengths

            if weights is not None:
                # weighted mean
                nll_loss *= weights
                nll_loss = nll_loss.sum()

            else:
                nll_loss = nll_loss.mean()
        else:
            nll_loss = F.nll_loss(pred, true, ignore_index = 0, reduction = "none")


        # Metrics
        metrics_dict = {}

        # If we use bayesian parameters and we're not doing predictions, calculate kld loss
        if self.bayesian and loss_reduction == "mean":
            metrics_dict["nll_loss"] = nll_loss.item()

            if neff is not None:
                kld_loss = self.parameter_kld() * (1 / neff) # distribute global loss onto the batch
            else:
                kld_loss = self.parameter_kld() * (1 / self.total_samples)

            if not self.multi_gpu:
                metrics_dict["kld_loss"] = kld_loss.item()
            total_loss = nll_loss + kld_loss
        else:
            total_loss = nll_loss

        # Regularize model parameters and distribute onto batch
        if self.l2_lambda > 0:
            l2_loss = 0
            for param in self.parameters():
                if param.requires_grad:
                    l2_loss += param.pow(2).sum()

            if neff is not None:
                l2_loss *= self.l2_lambda / neff # per sample l2 loss
            else:
                l2_loss *= self.l2_lambda / self.total_samples # per sample l2 loss

            if not self.multi_gpu:
                metrics_dict['l2_loss'] = l2_loss.item()

            total_loss += l2_loss

        return total_loss, metrics_dict

    def summary(self):
        num_params = sum(p.numel() for p in self.parameters())

        return (f"WaveNet summary:\n"
                f"  Input channels: {self.input_channels}\n"
                f"  Residual channels: {self.residual_channels}\n"
                f"  Output channels: {self.out_channels}\n"
                f"  Stacks: {self.stacks}\n"
                f"  Layers: {self.layers_per_stack} (max. {2**(self.layers_per_stack - 1)} dilation)\n"
                f"  Parameters: {num_params:,}\n"
                f"  Bayesian: {self.bayesian}\n")

    def save(self, f):
        args_dict = {
            "input_channels": self.input_channels,
            "residual_channels": self.residual_channels,
            "out_channels": self.out_channels,
            "stacks": self.stacks,
            "layers_per_stack": self.layers_per_stack,
            "bias": self.bias,
            "dropout": self.dropout,
            "use_bayesian": self.bayesian,
            "total_samples": self.total_samples,
            "backwards": self.backwards,
        }

        torch.save({
            "name": "WaveNet",
            "state_dict": self.state_dict(),
            "args_dict": args_dict,
        }, f)

class WaveNetLayer(nn.Module):
    def __init__(self, channels, dropout, kernel_size, activation, dilation, causal = True, bias = True, bayesian = False):
        super().__init__()

        self.channels = channels
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.activation = activation
        self.dilation = dilation
        self.causal = causal
        self.bias = bias
        self.bayesian = bayesian
        self.dropout_layer = nn.Dropout(self.dropout)

        self.layer_norm_in = nn.LayerNorm(self.channels)
        self.layer_norm_out = nn.LayerNorm(self.channels)

        if self.causal:
            self.padding = (self.kernel_size - 1) * self.dilation
        else:
            self.padding = (self.kernel_size - 1) // 2 * self.dilation

        self.before_layer = NormConv(
            in_channels = self.channels,
            out_channels = self.channels,
            use_bayesian = self.bayesian,
            activation = self.activation,
            kernel_size = 1,
            dilation = 1,
            bias = self.bias
        )

        self.middle_layer = NormConv(
            in_channels = self.channels,
            out_channels = self.channels,
            use_bayesian = self.bayesian,
            activation = self.activation,
            kernel_size = self.kernel_size,
            dilation = self.dilation,
            bias = self.bias
        )

        self.after_layer = NormConv(
            in_channels = self.channels,
            out_channels = self.channels,
            use_bayesian = self.bayesian,
            activation = self.activation,
            kernel_size = 1,
            dilation = 1,
            bias = self.bias
        )

    def _layer_norm(self, x, layer):
        return layer(x.permute(0, 2, 1)).permute(0, 2, 1)

    def forward(self, x):
        x = self._layer_norm(x, self.layer_norm_in)
        x = F.pad(x, (self.padding, 0))
        x = self.before_layer(x)
        x = self.middle_layer(x)
        x = self.after_layer(x)
        x = self.dropout_layer(x)
        x = self._layer_norm(x, self.layer_norm_out)
        return x

    def parameter_kld(self):
        kld = layer_kld(self.before_layer.layer)
        kld += layer_kld(self.middle_layer.layer)
        kld += layer_kld(self.after_layer.layer)
        return kld

class WaveNetStack(nn.Module):
    def __init__(self, dilations, channels, dropout, kernel_size, activation, causal = True, bias = True, bayesian = False):
        super().__init__()

        self.dilations = dilations
        self.channels = channels
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.causal = causal
        self.bias = bias
        self.bayesian = bayesian
        self.activation = activation

        self.layers = nn.ModuleList()
        for d in dilations:
            layer = WaveNetLayer(
                channels = self.channels,
                dropout = self.dropout,
                kernel_size = self.kernel_size,
                activation = self.activation,
                dilation = d,
                causal = self.causal,
                bias = self.bias,
                bayesian = self.bayesian
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x += layer(x)
        return x

    def parameter_kld(self):
        kld = 0
        for layer in self.layers:
            kld += layer.parameter_kld()

        return kld
