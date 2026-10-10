"""Phase-free confidence-gated attenuation model for the H-shield study."""
from __future__ import annotations

import torch
from torch import nn

from model_e import CompactAttenuationOnlyDereverb16k


class CompactDereverbHShield(CompactAttenuationOnlyDereverb16k):
    """Bounded attenuation with a learned confidence gate and no phase head.

    ``g = 1 - 0.5 * sigmoid(confidence_logit) * sigmoid(gain_logit)``.
    Thus every time-frequency bin is either passed through or attenuated;
    the model cannot amplify or phase-rotate the input.
    """

    MAX_ATTENUATION = 0.5
    GAIN_INIT_SEED = 2026101010
    GAIN_INIT_LOGIT = -3.0
    CONFIDENCE_INIT_LOGIT = 0.0
    GAIN_WEIGHT_STD = 1e-4
    CONFIDENCE_WEIGHT_STD = 1e-4

    def __init__(self, n_fft: int = 512, hop_length: int = 128,
                 base_channels: int = 16):
        super().__init__(n_fft=n_fft, hop_length=hop_length,
                         base_channels=base_channels)
        if self.mask_head.out_channels != 2:
            raise RuntimeError("H-shield expects exactly gain and confidence channels")
        with torch.no_grad():
            generator = torch.Generator(device="cpu").manual_seed(self.GAIN_INIT_SEED)
            for channel, std in ((0, self.GAIN_WEIGHT_STD),
                                 (1, self.CONFIDENCE_WEIGHT_STD)):
                weight = torch.empty_like(self.mask_head.weight[channel], device="cpu")
                nn.init.normal_(weight, mean=0.0, std=std, generator=generator)
                self.mask_head.weight[channel].copy_(weight.to(
                    device=self.mask_head.weight.device,
                    dtype=self.mask_head.weight.dtype))
            self.mask_head.bias[0] = self.GAIN_INIT_LOGIT
            self.mask_head.bias[1] = self.CONFIDENCE_INIT_LOGIT

    def forward_with_mask(self, audio: torch.Tensor
                          ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if audio.ndim != 2:
            raise ValueError(f"expected [batch, samples], got {tuple(audio.shape)}")
        length = audio.shape[-1]
        if length == 0:
            raise ValueError("audio must not be empty")
        window = self.window.to(dtype=audio.dtype, device=audio.device)
        spec = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window=window, center=True, return_complex=True)
        features = torch.stack((spec.real, spec.imag), dim=1)
        logits = self._predict_mask(features)
        attenuation = torch.sigmoid(logits[:, 0])
        confidence = torch.sigmoid(logits[:, 1])
        gain = 1.0 - self.MAX_ATTENUATION * confidence * attenuation
        enhanced = torch.istft(spec * gain, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.n_fft, window=window,
            center=True, length=length)
        return enhanced, gain, confidence

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.forward_with_mask(audio)[0]
