"""
TSFA model components for Two-Stage Feature Alignment.
Based on: "Universal Open-Set Domain Adaptation for Time-Series via SDE
and Optimal Transport" (TNNLS 2026).

Components:
    - STFTEncoder: STFT-based frequency feature extractor
    - TimeCNN: Time-domain CNN branch (reuses standard CNN pattern)
    - TFFeatureExtractor: Dual time-frequency encoder
    - NeuralSDE: Stochastic differential equation for global alignment
    - OSAM: Open-Set Alignment Module using class-aware optimal transport
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class STFTEncoder(nn.Module):
    """Extract frequency features via Short-Time Fourier Transform.

    STFT provides time-frequency representation (unlike RAINCOAT's global FFT).
    We take magnitude of STFT, then pass through conv layers to get a fixed-dim feature.
    """
    def __init__(self, input_channels, sequence_len, n_fft=64, hop_length=16,
                 out_channels=64, features_len=1):
        super(STFTEncoder, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # STFT output: (batch, channels, freq_bins, time_frames)
        # freq_bins = n_fft // 2 + 1
        freq_bins = n_fft // 2 + 1

        # Conv2d to process the spectrogram
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = out_channels

    def forward(self, x):
        """x: (batch, channels, seq_len)"""
        B, C, L = x.shape
        # Compute STFT per channel
        # Use a Hann window for better frequency resolution
        window = torch.hann_window(self.n_fft, device=x.device)
        specs = []
        for c in range(C):
            # stft returns complex tensor (B, freq_bins, time_frames)
            s = torch.stft(x[:, c, :], n_fft=self.n_fft, hop_length=self.hop_length,
                           window=window, return_complex=True)
            specs.append(s.abs())  # magnitude spectrogram
        # Stack channels: (B, C, freq_bins, time_frames)
        spec = torch.stack(specs, dim=1)

        # Conv2d feature extraction
        h = self.conv1(spec)
        h = self.conv2(h)
        h = self.pool(h)  # (B, out_channels, 1, 1)
        return h.view(B, -1)  # (B, out_channels)


class TFFeatureExtractor(nn.Module):
    """Dual time-frequency feature extractor for TSFA.

    Concatenates time-domain CNN features with STFT frequency features.
    """
    def __init__(self, time_backbone, configs):
        super(TFFeatureExtractor, self).__init__()
        self.time_encoder = time_backbone(configs)
        self.time_out_dim = configs.feat_dim

        # STFT encoder
        stft_n_fft = min(64, configs.sequence_len)
        stft_hop = max(stft_n_fft // 4, 1)
        stft_out = configs.final_out_channels
        self.freq_encoder = STFTEncoder(
            input_channels=configs.input_channels,
            sequence_len=configs.sequence_len,
            n_fft=stft_n_fft,
            hop_length=stft_hop,
            out_channels=stft_out,
        )
        self.freq_out_dim = stft_out
        self.out_dim = self.time_out_dim + self.freq_out_dim

    def forward(self, x):
        """Returns concatenated time + frequency features."""
        f_time = self.time_encoder(x)       # (B, time_out_dim)
        f_freq = self.freq_encoder(x)       # (B, freq_out_dim)
        return torch.cat([f_time, f_freq], dim=1)  # (B, out_dim)


class NeuralSDE(nn.Module):
    """Neural SDE for global domain alignment.

    Models the feature transformation as a stochastic process:
        dZ = f(Z, t)dt + g(Z, t)dW

    where f is the drift network and g is the diffusion network.
    We use Euler-Maruyama discretization over T steps.

    The SDE maps features from both domains into a shared latent space,
    encouraging domain-invariant representations.
    """
    def __init__(self, feat_dim, hidden_dim=128, num_steps=10):
        super(NeuralSDE, self).__init__()
        self.num_steps = num_steps
        self.dt = 1.0 / num_steps

        # Drift network f(z, t)
        self.drift = nn.Sequential(
            nn.Linear(feat_dim + 1, hidden_dim),  # +1 for time embedding
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )

        # Diffusion network g(z, t) -> scalar diffusion coefficient
        self.diffusion = nn.Sequential(
            nn.Linear(feat_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
            nn.Softplus(),  # ensure positive diffusion
        )

        self.out_dim = feat_dim

    def forward(self, z, return_trajectory=False):
        """Euler-Maruyama integration of the SDE.

        Args:
            z: (B, feat_dim) initial features
            return_trajectory: if True, return all intermediate states
        Returns:
            z_T: (B, feat_dim) final features
        """
        B = z.shape[0]
        trajectory = [z] if return_trajectory else None

        for step in range(self.num_steps):
            t = torch.full((B, 1), step * self.dt, device=z.device)
            zt = torch.cat([z, t], dim=1)

            drift = self.drift(zt)
            diffusion = self.diffusion(zt)

            # Euler-Maruyama: z_{t+1} = z_t + f(z,t)*dt + g(z,t)*sqrt(dt)*dW
            noise = torch.randn_like(z)
            z = z + drift * self.dt + diffusion * (self.dt ** 0.5) * noise

            if return_trajectory:
                trajectory.append(z)

        if return_trajectory:
            return z, trajectory
        return z

    def kl_divergence_loss(self, z_source, z_target):
        """Approximate KL divergence between SDE outputs for global alignment.

        We match the first two moments (mean and variance) of the
        transformed source and target distributions.
        """
        z_s = self.forward(z_source)
        z_t = self.forward(z_target)

        # Match means
        mean_loss = F.mse_loss(z_s.mean(dim=0), z_t.mean(dim=0))

        # Match variances
        var_s = z_s.var(dim=0)
        var_t = z_t.var(dim=0)
        var_loss = F.mse_loss(var_s, var_t)

        return mean_loss + var_loss, z_s, z_t
