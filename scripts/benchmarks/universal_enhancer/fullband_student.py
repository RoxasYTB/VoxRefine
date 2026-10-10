"""Compact full-band complex-STFT residual enhancer for GTX 1050-class GPUs."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for group in (8, 4, 2, 1):
        if channels % group == 0:
            return group
    return 1


class Block(nn.Module):
    def __init__(self, incoming: int, outgoing: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1),
            nn.GroupNorm(_groups(outgoing), outgoing), nn.PReLU(outgoing),
            nn.Conv2d(outgoing, outgoing, 3, padding=1),
            nn.GroupNorm(_groups(outgoing), outgoing), nn.PReLU(outgoing),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class FullbandStudent(nn.Module):
    """Offline 48 kHz speech restoration, residual in complex STFT space."""

    def __init__(self, n_fft: int = 1024, hop_length: int = 256, base_channels: int = 8):
        super().__init__()
        self.n_fft, self.hop_length = int(n_fft), int(hop_length)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        c1, c2, c3, cb = base_channels, base_channels * 2, base_channels * 3, base_channels * 4
        self.e1, self.d1 = Block(2, c1), nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.e2, self.d2 = Block(c2, c2), nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.e3, self.d3 = Block(c3, c3), nn.Conv2d(c3, cb, 3, stride=2, padding=1)
        self.mid = Block(cb, cb)
        self.u3, self.o3 = nn.Conv2d(cb, c3, 1), Block(c3, c3)
        self.u2, self.o2 = nn.Conv2d(c3, c2, 1), Block(c2, c2)
        self.u1, self.o1 = nn.Conv2d(c2, c1, 1), Block(c1, c1)
        self.head = nn.Conv2d(c1, 2, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor, projection: nn.Module) -> torch.Tensor:
        return projection(F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)) + skip

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError(f"expected [batch, samples], got {tuple(waveform.shape)}")
        n = waveform.shape[-1]
        spec = torch.stft(waveform, self.n_fft, self.hop_length, self.n_fft,
                          self.window.to(waveform), center=True, return_complex=True)
        feat = torch.stack((spec.real, spec.imag), dim=1)
        a = self.e1(feat)
        b = self.e2(self.d1(a))
        c = self.e3(self.d2(b))
        z = self.mid(self.d3(c))
        x = self.o3(self._up(z, c, self.u3))
        x = self.o2(self._up(x, b, self.u2))
        x = self.o1(self._up(x, a, self.u1))
        delta = self.head(x)
        restored = spec + torch.complex(delta[:, 0], delta[:, 1])
        return torch.istft(restored, self.n_fft, self.hop_length, self.n_fft,
                           self.window.to(waveform), center=True, length=n)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def stft_l1(pred: torch.Tensor, target: torch.Tensor, n_fft: int, hop: int,
            window: torch.Tensor) -> torch.Tensor:
    p = torch.stft(pred, n_fft, hop, n_fft, window.to(pred), center=True, return_complex=True)
    t = torch.stft(target, n_fft, hop, n_fft, window.to(target), center=True, return_complex=True)
    return (p.abs() - t.abs()).abs().mean()
