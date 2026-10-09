"""Small deterministic complex-STFT residual model for the 16 kHz proof."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for candidate in (8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CompactDereverb16k(nn.Module):
    """Offline 16 kHz residual estimator; fixed-input inference is deterministic."""

    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 128,
        base_channels: int = 16,
    ):
        super().__init__()
        if n_fft <= 0 or hop_length <= 0 or base_channels <= 0:
            raise ValueError("FFT, hop, and base channel counts must be positive")
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        c1, c2, c3, c4, cb = (
            base_channels,
            base_channels * 2,
            base_channels * 3,
            base_channels * 4,
            base_channels * 6,
        )
        self.enc1 = ConvBlock(2, c1)
        self.down1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.enc2 = ConvBlock(c2, c2)
        self.down2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.enc3 = ConvBlock(c3, c3)
        self.down3 = nn.Conv2d(c3, c4, 3, stride=2, padding=1)
        self.enc4 = ConvBlock(c4, c4)
        self.down4 = nn.Conv2d(c4, cb, 3, stride=2, padding=1)
        self.bottleneck = ConvBlock(cb, cb)

        self.proj4 = nn.Conv2d(cb, c4, 1)
        self.dec4 = ConvBlock(c4, c4)
        self.proj3 = nn.Conv2d(c4, c3, 1)
        self.dec3 = ConvBlock(c3, c3)
        self.proj2 = nn.Conv2d(c3, c2, 1)
        self.dec2 = ConvBlock(c2, c2)
        self.proj1 = nn.Conv2d(c2, c1, 1)
        self.dec1 = ConvBlock(c1, c1)
        self.residual_head = nn.Conv2d(c1, 2, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def _up_add(self, x: torch.Tensor, skip: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return projection(x) + skip

    def _predict_residual(self, features: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(features)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        mid = self.bottleneck(self.down4(e4))
        d4 = self.dec4(self._up_add(mid, e4, self.proj4))
        d3 = self.dec3(self._up_add(d4, e3, self.proj3))
        d2 = self.dec2(self._up_add(d3, e2, self.proj2))
        d1 = self.dec1(self._up_add(d2, e1, self.proj1))
        return self.residual_head(d1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 2:
            raise ValueError(f"expected [batch, samples], got {tuple(audio.shape)}")
        original_length = audio.shape[-1]
        if original_length == 0:
            raise ValueError("audio must not be empty")
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(dtype=audio.dtype, device=audio.device),
            center=True,
            return_complex=True,
        )
        features = torch.stack((spec.real, spec.imag), dim=1)
        residual = self._predict_residual(features)
        enhanced_spec = spec + torch.complex(residual[:, 0], residual[:, 1])
        return torch.istft(
            enhanced_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window.to(dtype=audio.dtype, device=audio.device),
            center=True,
            length=original_length,
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
