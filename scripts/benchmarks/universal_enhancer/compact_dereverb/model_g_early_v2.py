"""G-early-v2 gain head: bounded attenuation with a less saturated start.

This module is isolated from model_e.py so existing E experiments and
checkpoints retain their original behavior.
"""
from __future__ import annotations

import torch

from model_e import CompactAttenuationOnlyDereverb16k


class CompactAttenuationOnlyDereverbG2(CompactAttenuationOnlyDereverb16k):
    """E U-Net with gain `1 - 0.5*sigmoid(logit)` and initialization `logit=-3`.

    The gain is strictly bounded to (0.5, 1): it cannot amplify any STFT bin,
    and permits up to 6.02 dB of amplitude attenuation. Only the gain mapping
    and its initial bias differ from E; the architecture and phase path match.
    """

    MAX_ATTENUATION = 0.5
    INITIAL_GAIN_LOGIT = -3.0
    GAIN_WEIGHT_STD = 1e-4
    GAIN_INIT_SEED = 2026101002

    def __init__(self, n_fft: int = 512, hop_length: int = 128,
                 base_channels: int = 16):
        super().__init__(n_fft=n_fft, hop_length=hop_length,
                         base_channels=base_channels)
        with torch.no_grad():
            generator = torch.Generator(device="cpu").manual_seed(self.GAIN_INIT_SEED)
            gain_weight = torch.empty_like(self.mask_head.weight[0], device="cpu")
            torch.nn.init.normal_(gain_weight, mean=0.0, std=self.GAIN_WEIGHT_STD,
                                  generator=generator)
            self.mask_head.weight[0].copy_(gain_weight.to(
                device=self.mask_head.weight.device, dtype=self.mask_head.weight.dtype))
            self.mask_head.weight[1].zero_()
            self.mask_head.bias[0] = self.INITIAL_GAIN_LOGIT
            self.mask_head.bias[1] = 0.0

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
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
        gain = 1.0 - self.MAX_ATTENUATION * torch.sigmoid(logits[:, 0])
        phase = torch.pi * torch.tanh(logits[:, 1])
        enhanced_spec = spec * torch.polar(gain, phase)
        return torch.istft(enhanced_spec, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.n_fft, window=window,
            center=True, length=length)


def initial_gain_diagnostics() -> dict[str, float]:
    """Return closed-form initialization checks without importing model weights."""
    import math

    logit = CompactAttenuationOnlyDereverbG2.INITIAL_GAIN_LOGIT
    sigmoid = 1.0 / (1.0 + math.exp(-logit))
    gain = 1.0 - CompactAttenuationOnlyDereverbG2.MAX_ATTENUATION * sigmoid
    derivative_magnitude = CompactAttenuationOnlyDereverbG2.MAX_ATTENUATION * sigmoid * (1.0 - sigmoid)
    gain_for_3db_energy = math.sqrt(10.0 ** (-3.0 / 10.0))
    sigmoid_for_3db = (1.0 - gain_for_3db_energy) / CompactAttenuationOnlyDereverbG2.MAX_ATTENUATION
    logit_for_3db = math.log(sigmoid_for_3db / (1.0 - sigmoid_for_3db))
    return {
        "initial_logit": logit,
        "initial_gain": gain,
        "initial_amplitude_attenuation_db": -20.0 * math.log10(gain),
        "initial_abs_gain_derivative": derivative_magnitude,
        "old_sigmoid_plus5_abs_derivative": (1.0 / (1.0 + math.exp(-5.0))) *
            (1.0 - 1.0 / (1.0 + math.exp(-5.0))),
        "gain_for_3db_energy_reduction": gain_for_3db_energy,
        "logit_for_3db_energy_reduction": logit_for_3db,
        "max_amplitude_attenuation_db": -20.0 * math.log10(1.0 - 0.5),
    }
