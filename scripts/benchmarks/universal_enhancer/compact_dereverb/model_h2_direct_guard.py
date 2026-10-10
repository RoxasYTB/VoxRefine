"""Phase-free attenuation model for the H2 direct-guard study."""
from __future__ import annotations

import torch
from torch import nn

from model_e import CompactAttenuationOnlyDereverb16k


class CompactDereverbH2DirectGuard(CompactAttenuationOnlyDereverb16k):
    """Bounded gain-only mask, with the phase head removed entirely."""

    MAX_ATTENUATION = 0.5
    INIT_SEED = 2026101011
    INIT_LOGIT = -3.0
    WEIGHT_STD = 1e-4

    def __init__(self, n_fft: int = 512, hop_length: int = 128,
                 base_channels: int = 16):
        super().__init__(n_fft=n_fft, hop_length=hop_length,
                         base_channels=base_channels)
        with torch.no_grad():
            generator = torch.Generator(device="cpu").manual_seed(self.INIT_SEED)
            weight = torch.empty_like(self.mask_head.weight[0], device="cpu")
            nn.init.normal_(weight, mean=0.0, std=self.WEIGHT_STD,
                            generator=generator)
            self.mask_head.weight[0].copy_(weight.to(
                device=self.mask_head.weight.device,
                dtype=self.mask_head.weight.dtype))
            self.mask_head.bias[0] = self.INIT_LOGIT
        self.mask_head.weight.register_hook(lambda grad: torch.cat(
            (grad[:1], torch.zeros_like(grad[1:])), dim=0))
        self.mask_head.bias.register_hook(lambda grad: torch.cat(
            (grad[:1], torch.zeros_like(grad[1:])), dim=0))

    def forward_with_gain(self, audio: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        if audio.ndim != 2:
            raise ValueError(f"expected [batch, samples], got {tuple(audio.shape)}")
        length = audio.shape[-1]
        if length == 0:
            raise ValueError("audio must not be empty")
        window = self.window.to(dtype=audio.dtype, device=audio.device)
        spec = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=window, center=True,
            return_complex=True)
        features = torch.stack((spec.real, spec.imag), dim=1)
        logits = self._predict_mask(features)
        gain = 1.0 - self.MAX_ATTENUATION * torch.sigmoid(logits[:, 0])
        enhanced = torch.istft(spec * gain, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.n_fft, window=window,
            center=True, length=length)
        return enhanced, gain

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.forward_with_gain(audio)[0]
