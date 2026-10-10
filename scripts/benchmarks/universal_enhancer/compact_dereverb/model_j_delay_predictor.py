"""Causal delayed complex-spectrum residual predictor for study J."""
from __future__ import annotations

import torch
from torch import nn


class JDelayPredictor16k(nn.Module):
    """Predict a late complex residual from history, never the current frame."""

    def __init__(self, n_fft: int = 512, hop_length: int = 128,
                 delay_frames: int = 4, hidden_size: int = 128):
        super().__init__()
        if min(n_fft, hop_length, delay_frames, hidden_size) <= 0:
            raise ValueError("FFT, hop, delay, and hidden size must be positive")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.delay_frames = delay_frames
        self.hidden_size = hidden_size
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        bins = n_fft // 2 + 1
        # Bias-free layers ensure an all-zero input produces an exactly zero
        # residual after training too, rather than hallucinated room output.
        self.predictor = nn.GRU(2 * bins, hidden_size, num_layers=1,
                                batch_first=True, bidirectional=False, bias=False)
        self.residual_head = nn.Linear(hidden_size, 2 * bins, bias=False)
        nn.init.zeros_(self.residual_head.weight)

    def forward_with_residual(self, audio: torch.Tensor
                              ) -> tuple[torch.Tensor, torch.Tensor]:
        if audio.ndim != 2 or audio.shape[-1] == 0:
            raise ValueError("expected nonempty [batch, samples] audio")
        length = audio.shape[-1]
        window = self.window.to(device=audio.device, dtype=audio.dtype)
        spec = torch.stft(audio, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.n_fft, window=window,
            center=True, return_complex=True)
        # [B,T,2F]. The GRU state at t-D can only use observations through t-D.
        features = torch.cat((spec.real, spec.imag), dim=1).transpose(1, 2)
        states, _ = self.predictor(features)
        delayed = torch.zeros_like(states)
        if states.shape[1] > self.delay_frames:
            delayed[:, self.delay_frames:] = states[:, :-self.delay_frames]
        residual_ri = self.residual_head(delayed).transpose(1, 2)
        residual = torch.complex(residual_ri[:, :spec.shape[1]],
                                 residual_ri[:, spec.shape[1]:])
        residual_wave = torch.istft(residual, n_fft=self.n_fft,
            hop_length=self.hop_length, win_length=self.n_fft, window=window,
            center=True, length=length)
        # Subtract the synthesized residual from the original samples directly.
        # This preserves bit-exact identity when the zero-initialized head emits 0.
        enhanced = audio - residual_wave
        return enhanced, residual_wave

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.forward_with_residual(audio)[0]
