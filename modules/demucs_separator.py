"""
demucs_separator.py
Thin wrapper around local Demucs model for vocal/accompaniment separation.
If separation is disabled in config, the module simply passes through the
input audio (optionally converting to mono).
"""

import logging
import os
import tempfile
from typing import Tuple, Optional

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

# Try to import Demucs; if not available, we will still allow the class
# to be instantiated but separation will be disabled.
try:
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    _DEMUCS_AVAILABLE = True
except Exception as e:  # pragma: no cover
    logger.warning(f"Demucs import failed: {e}")
    _DEMUCS_AVAILABLE = False


class DemucsSeparator:
    """
    Parameters
    ----------
    config : dict
        The configuration dictionary, expecting a 'source_separation' key.
    device : str
        torch device string, e.g., 'cpu' or 'cuda'.
    """

    def __init__(self, config: dict, device: str = "cpu"):
        sep_cfg = config.get("source_separation", {})
        self.enable = bool(sep_cfg.get("enable", False))
        self.model_name = sep_cfg.get("model", "demucs_quantized")
        self.device = device
        self._model = None

        if self.enable and _DEMUCS_AVAILABLE:
            logger.info(f"Loading Demucs model '{self.model_name}' on {device}")
            self._model = get_model(self.model_name)
            self._model.to(device)
        else:
            # Either disabled or Demucs not available -> passthrough mode
            self.enable = False
            logger.info(
                "Demucs separator disabled (will passthrough audio). "
                "If you intended to enable separation, verify that "
                "'source_separation.enable' is true and Demucs is importable."
            )

    def separate(
        self, waveform: np.ndarray, sample_rate: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Separate vocals from accompaniment.

        Parameters
        ----------
        waveform : np.ndarray
            Input audio, shape (samples,) or (channels, samples).
        sample_rate : int
            Sampling rate of the input audio.

        Returns
        -------
        vocals : np.ndarray
            Mono vocal waveform (float32).
        accompaniment : Optional[np.ndarray]
            Mono accompaniment waveform (float32) if separation was performed
            and accompaniment could be extracted; otherwise None.
        """
        # If separation disabled or model not loaded, just return mono waveform.
        if not self.enable or self._model is None:
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            return waveform.astype(np.float32), None

        # ---------- Actual Demucs separation ----------
        # Demucs expects a torch tensor of shape [1, C, T] (batch=1).
        # Ensure we have at least stereo; if mono, duplicate to two channels.
        if waveform.ndim == 1:
            waveform = np.stack([waveform, waveform], axis=0)  # [2, T]
        elif waveform.ndim == 2 and waveform.shape[0] == 1:
            waveform = np.repeat(waveform, 2, axis=0)          # [2, T]
        # Convert to float32 tensor.
        waveform_tensor = torch.from_numpy(waveform).to(
            dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # [1, 2, T]

        with torch.no_grad():
            # apply_model returns [1, num_sources, C, T]
            sources = apply_model(
                self._model,
                waveform_tensor,
                device=self.device,
                progress=False,
            )[0]  # Remove batch dim -> [num_sources, 2, T]

        # Determine which source corresponds to vocals.
        # Many Demucs models expose .sources attribute (list of strings).
        vocal_wav: Optional[np.ndarray] = None
        accomp_wav: Optional[np.ndarray] = None

        src_names = getattr(self._model, "sources", None)
        if src_names is not None:
            try:
                vocal_idx = src_names.index("vocals")
                vocal_wav = sources[vocal_idx].cpu().numpy()
                # Accompaniment: sum of all other sources.
                other_idx = [i for i, name in enumerate(src_names) if name != "vocals"]
                if other_idx:
                    accomp_wav = sources[other_idx].sum(axis=0).cpu().numpy()
                else:
                    accomp_wav = None
            except ValueError:
                # No "vocals" in source names; fallback heuristics.
                pass
        else:
            # No source names; assume first half are vocals, second half accompaniment
            # (common for 2‑stem models). This is a heuristic.
            n_src = sources.shape[0]
            half = n_src // 2
            if half > 0:
                vocal_wav = sources[:half].sum(axis=0).cpu().numpy()
                if n_src > half:
                    accomp_wav = sources[half:].sum(axis=0).cpu().numpy()
                else:
                    accomp_wav = None
            else:
                # Should not happen, but fallback to treating everything as vocals.
                vocal_wav = sources.sum(axis=0).cpu().numpy()
                accomp_wav = None

        # If for some reason we didn't get vocals, fallback to mixture.
        if vocal_wav is None:
            vocal_wav = waveform_tensor.squeeze(0).cpu().numpy().mean(axis=0)
            accomp_wav = None

        # Ensure mono and float32.
        if vocal_wav.ndim > 1:
            vocal_wav = vocal_wav.mean(axis=0)
        vocal_wav = vocal_wav.astype(np.float32)

        if accomp_wav is not None:
            if accomp_wav.ndim > 1:
                accomp_wav = accomp_wav.mean(axis=0)
            accomp_wav = accomp_wav.astype(np.float32)

        logger.debug(
            f"Demucs separation completed: vocal shape {vocal_wav.shape}, "
            f"accompaniment {'present' if accomp_wav is not None else 'absent'}"
        )
        return vocal_wav, accomp_wav


def get_separator(config: dict, device: str = "cpu") -> DemucsSeparator:
    """Factory function to create a DemucsSeparator instance."""
    return DemucsSeparator(config, device)