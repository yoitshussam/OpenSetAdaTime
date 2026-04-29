"""
Raincoat model components for time-frequency domain adaptation.
Adapted from: He et al., "Domain Adaptation for Time Series Under Feature
and Label Shifts", ICML 2023.

Components:
    - SpectralConv1d: 1D Fourier layer (FFT -> linear -> iFFT)
    - RaincoatCNN: Time-domain CNN branch
    - TFEncoder: Dual time-frequency encoder
    - TFDecoder: Time-frequency decoder
    - RaincoatClassifier: Temperature-scaled linear classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """1D Fourier layer: FFT -> linear transform in spectral domain -> inverse FFT."""
    def __init__(self, in_channels, out_channels, modes1, fl=128):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x = torch.cos(x)
        x_ft = torch.fft.rfft(x, norm='ortho')
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-1) // 2 + 1,
            device=x.device, dtype=torch.cfloat
        )
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)
        r = out_ft[:, :, :self.modes1].abs()
        p = out_ft[:, :, :self.modes1].angle()
        return torch.concat([r, p], -1), out_ft


class RaincoatCNN(nn.Module):
    """Time-domain CNN branch (conv layers + adaptive pool)."""
    def __init__(self, configs):
        super(RaincoatCNN, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels,
                      kernel_size=configs.kernel_size, stride=configs.stride,
                      bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.final_out_channels,
                      kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block3(x)
        x = self.adaptive_pool(x)
        return x.reshape(x.shape[0], -1)


class TFEncoder(nn.Module):
    """Dual time-frequency encoder."""
    def __init__(self, configs):
        super(TFEncoder, self).__init__()
        self.modes1 = configs.fourier_modes
        self.width = configs.input_channels
        self.length = configs.sequence_len
        self.freq_feature = SpectralConv1d(self.width, self.width, self.modes1, self.length)
        self.bn_freq = nn.BatchNorm1d(configs.fourier_modes * 2)
        self.cnn = RaincoatCNN(configs)
        self.avg = nn.Conv1d(self.width, 1, kernel_size=3,
                             stride=configs.stride, bias=False, padding=(3 // 2))

    def forward(self, x):
        ef, out_ft = self.freq_feature(x)
        ef = F.relu(self.bn_freq(self.avg(ef).squeeze()))
        et = self.cnn(x)
        f = torch.concat([ef, et], -1)
        return F.normalize(f), out_ft


class TFDecoder(nn.Module):
    """Time-frequency decoder."""
    def __init__(self, configs):
        super(TFDecoder, self).__init__()
        self.input_channels = configs.input_channels
        self.sequence_len = configs.sequence_len
        self.bn1 = nn.BatchNorm1d(self.input_channels, self.sequence_len)
        self.bn2 = nn.BatchNorm1d(self.input_channels, self.sequence_len)
        self.convT = nn.ConvTranspose1d(
            configs.final_out_channels, self.sequence_len, self.input_channels, stride=1
        )
        self.modes = configs.fourier_modes

    def forward(self, f, out_ft):
        x_low = self.bn1(torch.fft.irfft(out_ft, n=self.sequence_len))
        et = f[:, self.modes * 2:]
        x_high = F.relu(self.bn2(self.convT(et.unsqueeze(2)).permute(0, 2, 1)))
        return x_low + x_high


class RaincoatClassifier(nn.Module):
    """Classifier head with temperature scaling."""
    def __init__(self, configs):
        super(RaincoatClassifier, self).__init__()
        self.logits = nn.Linear(configs.out_dim, configs.num_classes, bias=False)
        self.tmp = 0.1

    def forward(self, x):
        return self.logits(x) / self.tmp
