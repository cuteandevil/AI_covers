"""
Audio utility functions.
"""
import torch
import torchaudio
import numpy as np
import librosa

def load_wav(path: str, target_sr: int = 24000) -> torch.Tensor:
    """Load audio file and resample to target sample rate."""
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # convert to mono
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav

def save_wav(wav: torch.Tensor, path: str, sr: int = 24000):
    """Save waveform to file."""
    torchaudio.save(path, wav, sr)

def normalize_wav(wav: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Peak normalize waveform."""
    max_abs = wav.abs().max()
    if max_abs > eps:
        wav = wav / max_abs
    return wav