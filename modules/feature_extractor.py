"""
Feature extraction: F0, energy, speaker embedding.
"""
import torch
import torch.nn as nn
import numpy as np
import librosa
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, f0_config: dict, speaker_config: dict, device: str = "cpu"):
        self.device = device
        self.f0_method = f0_config.get("method", "crepe")
        self.f0_min = f0_config.get("f0_min", 50)
        self.f0_max = f0_config.get("f0_max", 500)
        self.hop_length = f0_config.get("hop_length", 160)
        # Speaker encoder placeholder (ECAPA-TDNN)
        self.speaker_encoder = self._load_speaker_encoder(speaker_config, device)

    def _load_speaker_encoder(self, config: dict, device: str):
        # In a real system, load a pretrained ECAPA-TDNN or similar.
        # Here we return a dummy projection network.
        embed_dim = config.get("embedding_dim", 256)
        logger.info("Initializing dummy speaker encoder (placeholder)")
        return nn.Linear(80, embed_dim).to(device)  # expects log-mel spec

    def extract(self, audio_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract F0, energy, and speaker embedding from audio.
        Returns:
            f0: Tensor of shape (T,)
            energy: Tensor of shape (T,)
            speaker_emb: Tensor of shape (D,)
        """
        logger.debug(f"Loading audio {audio_path}")
        wav, sr = librosa.load(audio_path, sr=None, mono=True)
        # Energy (RMS per frame)
        energy = librosa.feature.rms(y=wav, frame_length=2048, hop_length=self.hop_length)[0]
        energy = torch.from_numpy(energy).float().to(self.device)

        # F0 extraction
        if self.f0_method == "crepe":
            # Use crepe if available, else fallback to librosa's pyin
            try:
                import crepe
                timestamp, f0_confidence = crepe.predict(wav, sr, viterbi=True, step_size=self.hop_length*1000/sr)
                f0 = torch.from_numpy(f0_confidence).float().to(self.device)
            except ImportError:
                logger.warning("CREPE not installed, using librosa pyin")
                f0 = librosa.pyin(wav, fmin=self.f0_min, fmax=self.f0_max,
                                 sr=sr, frame_length=2048, hop_length=self.hop_length)[0]
                f0 = np.nan_to_num(f0)
                f0 = torch.from_numpy(f0).float().to(self.device)
        else:
            # Default to librosa pyin
            f0 = librosa.pyin(wav, fmin=self.f0_min, fmax=self.f0_max,
                              sr=sr, frame_length=2048, hop_length=self.hop_length)[0]
            f0 = np.nan_to_num(f0)
            f0 = torch.from_numpy(f0).float().to(self.device)

        # Speaker embedding from log-mel spectrogram
        mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=80,
                                             hop_length=self.hop_length)
        logmel = torch.from_numpy(librosa.power_to_db(mel)).float().to(self.device)
        # Average over time
        speaker_emb = self.speaker_encoder(logmel.mean(dim=1, keepdim=False).unsqueeze(0))
        speaker_emb = speaker_emb.squeeze(0)

        logger.debug(f"Extracted F0 shape {f0.shape}, energy {energy.shape}, speaker emb {speaker_emb.shape}")
        return f0, energy, speaker_emb