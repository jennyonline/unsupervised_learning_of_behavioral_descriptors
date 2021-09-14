import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn

from deepVAEHelpers.truncated_normal import TruncatedNormal


@torch.jit.script
def gaussian_analytical_kl(mu1, mu2, logsigma1, logsigma2):
    sigma1 = torch.exp(torch.tanh(logsigma1 / 8) * 8)
    sigma2 = torch.exp(torch.tanh(logsigma2 / 8) * 8)
    return (
        -0.5
        + logsigma2
        - logsigma1
        + 0.5 * (sigma1 ** 2 + (mu1 - mu2) ** 2) / (sigma2 ** 2)
    )


@torch.jit.script
def draw_gaussian_diag_samples(mu, logsigma):
    eps = torch.empty_like(mu).normal_(0.0, 1.0)
    sigma = torch.exp(torch.tanh(logsigma / 8) * 8)
    return sigma * eps + mu


def get_conv(
    in_dim,
    out_dim,
    kernel_size,
    stride,
    padding,
    zero_bias=True,
    zero_weights=False,
    groups=1,
    scaled=False,
    spectral_norm=False
):
    conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, groups=groups)
    if zero_bias:
        conv.bias.data *= 0.0
    if zero_weights:
        conv.weight.data *= 0.0
    if spectral_norm:
        conv = torch.nn.utils.spectral_norm(conv)
    return conv


def get_3x3(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False, spectral_norm=False):
    return get_conv(in_dim, out_dim, 3, 1, 1, zero_bias, zero_weights, groups=groups, scaled=scaled, spectral_norm=spectral_norm)


def get_1x1(in_dim, out_dim, zero_bias=True, zero_weights=False, groups=1, scaled=False ,spectral_norm=False):
    return get_conv(in_dim, out_dim, 1, 1, 0, zero_bias, zero_weights, groups=groups, scaled=scaled, spectral_norm=spectral_norm)


def get_mixture_params(coeffs, eps=1e-8):
    ch = coeffs.size()[1]
    n_mix = ch // 3
    pi_logits = coeffs[:, :n_mix].permute(0, 2, 3, 4, 1)
    means = coeffs[:, n_mix : 2 * n_mix].permute(0, 2, 3, 4, 1)
    logsigmas = coeffs[:, 2 * n_mix:].permute(0,2, 3, 4, 1)
    sigmas = torch.exp(torch.tanh(logsigmas / 8) * 8)
    return pi_logits, means, sigmas


class DiscretizedMixture(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.width = H.width
        self.out_conv = get_conv(
            H.width, H.image_channels * H.num_mixtures * 3, kernel_size=1, stride=1, padding=0
        )

    def get_component_distributions(self, means, scale):
        raise NotImplementedError

    def get_mixture(self, px_z):
        pi_logits, means, scale = get_mixture_params(self.forward(px_z))

        batch_size = means.shape[0]
        data_shape = (batch_size, self.H.image_size, self.H.image_size, self.H.image_channels)

        means = means.reshape(-1, self.H.num_mixtures)
        scale = scale.reshape(-1, self.H.num_mixtures)
        pi_logits = pi_logits.reshape(-1, self.H.num_mixtures)
        component = self.get_component_distributions(means, scale)
        mix = D.Categorical(logits=pi_logits)
        mixture = D.MixtureSameFamily(mix, component)

        return mixture, data_shape

    def nll(self, px_z, x, eps=1e-8):
        mixture, data_shape = self.get_mixture(px_z)

        bin_size = (1 / self.H.std) / 2
        x_flat = x.reshape(-1)
        log_prob = (mixture.cdf(x_flat + bin_size) - mixture.cdf(x_flat - bin_size) + eps).log()
        return -log_prob.reshape(data_shape)

    def forward(self, px_z):
        xhat = self.out_conv(px_z)
        batch_size = xhat.shape[0]
        xhat = xhat.reshape(
            batch_size,
            self.H.num_mixtures * 3,
            self.H.image_channels,
            self.H.image_size,
            self.H.image_size,
        )
        xhat = xhat.permute(0, 1, 3, 4, 2)
        return xhat

    def sample(self, px_z, as_uint8=True):
        mixture, data_shape = self.get_mixture(px_z)
        samples = (mixture.sample().reshape(data_shape) * self.H.std) + self.H.mean
        if as_uint8:
            samples = torch.clamp(samples, 0, 255).data.cpu().numpy().astype(np.uint8)
        else:
            samples = samples.data.cpu().numpy()
        return samples


class DiscretizedTruncatedNormalMixture(DiscretizedMixture):
    def __init__(self, H):
        super().__init__(H)

    def get_component_distributions(self, means, scale):
        truncnorm = TruncatedNormal(
            means, scale, -self.H.mean / self.H.std, (255 - self.H.mean) / self.H.std
        )
        return truncnorm


class DiscretizedLogisticMixture(DiscretizedMixture):
    def __init__(self, H):
        super().__init__(H)

    def get_component_distributions(self, means, scale):
        base_distribution = D.Uniform(torch.zeros_like(means), torch.ones_like(means))
        transforms = [D.SigmoidTransform().inv, D.AffineTransform(loc=means, scale=scale)]
        logistic = D.TransformedDistribution(base_distribution, transforms)
        return logistic
