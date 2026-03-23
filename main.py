#!/usr/bin/env python
"""
Main entry point for AI Cover Generator.
Orchestrates the full pipeline: optional source separation → ASR → feature extraction →
multimodal fusion → adaptive learner → neural vocoder → quality monitoring → output.
"""

import argparse
import yaml
import logging
import os
import tempfile
import torch
import torchaudio
import soundfile as sf
import numpy as np
import torchaudio.functional as F
from pathlib import Path

# Import local modules
from modules.asr import ASRModule
from modules.feature_extractor import FeatureExtractor
from modules.adaptive_learner import AdaptiveLearner
from modules.neural_vocoder import NeuralVocoder
from modules.quality_monitor import QualityMonitor
from modules.demucs_separator import get_separator
from utils.logger import setup_logger


def load_config(config_path: str = "config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="AI Cover Generator")
    parser.add_argument("--input", type=str, required=True, help="Path to input audio (dry vocal or accompaniment)")
    parser.add_argument("--target_speaker", type=str, required=True, help="Identifier or audio file for target singing voice")
    parser.add_argument("--output", type=str, required=True, help="Path to save generated cover audio")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--save_accompaniment",
        action="store_true",
        help="If set, save the separated accompaniment (when source separation enabled).",
    )
    args = parser.parse_args()

    # Setup logging
    config = load_config(args.config)
    logger = setup_logger(config['logging'])
    logger.info("Starting AI Cover Generator")
    logger.debug(f"Config: {config}")

    # Determine device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load input audio
    logger.info(f"Loading input audio from {args.input}")
    waveform, sr = sf.read(args.input, always_2d=False)  # may be mono or multi-channel
    if waveform.ndim > 1:
        # Convert to mono for separation and downstream processing
        waveform = waveform.mean(axis=1)
    waveform = waveform.astype(np.float32)

    # Optional source separation (Demucs)
    separator = get_separator(config, device=str(device))
    vocals_wav, accompaniment_wav = separator.separate(waveform, sr)

    # If separation produced accompaniment and user wants to save it, write to file
    accompaniment_path = None
    if accompaniment_wav is not None and args.save_accompaniment:
        accompaniment_path = os.path.join(
            os.path.dirname(os.path.abspath(args.output)),
            f"accompaniment_{os.path.basename(args.output)}",
        )
        sf.write(accompaniment_path, accompaniment_wav, sr)
        logger.info(f"Accompaniment saved to: {accompaniment_path}")

    # Prepare audio for ASR/feature extraction: use separated vocals if available, else original mixture
    if vocals_wav is not None:
        # Write vocals to a temporary file (ASR and FeatureExtractor expect file paths)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_vocals:
            sf.write(tmp_vocals.name, vocals_wav, sr)
            vocals_path = tmp_vocals.name
        logger.info(f"Using separated vocals (temp file: {vocals_path}) for ASR and feature extraction")
    else:
        vocals_path = args.input
        logger.info("No separation performed; using original input audio for ASR and feature extraction")

    # Initialize modules
    asr = ASRModule(config['asr'], device=str(device))
    feat_extractor = FeatureExtractor(config['f0_extractor'], config['speaker_encoder'], device=str(device))
    adaptive = AdaptiveLearner(config['adaptive_learner'], device=str(device))
    vocoder = NeuralVocoder(config['vocoder'], device=str(device))
    quality_monitor = (
        QualityMonitor(config['quality_monitor'])
        if config['quality_monitor']['enable']
        else None
    )

    # Step 1: ASR → lyric text
    logger.info("Extracting lyrics via ASR...")
    lyrics = asr.transcribe(vocals_path)
    logger.debug(f"Lyrics: {lyrics[:100]}...")

    # Step 2: Extract F0, energy, speaker embedding from input audio (vocals)
    logger.info("Extracting acoustic features...")
    f0, energy, speaker_emb = feat_extractor.extract(vocals_path)
    logger.debug(f"F0 shape: {f0.shape}, Energy shape: {energy.shape}, Speaker emb shape: {speaker_emb.shape}")

    # Step 3: Prepare target speaker representation (few-shot adaptation)
    logger.info("Preparing target speaker representation...")
    target_repr = adaptive.get_speaker_representation(args.target_speaker)
    logger.debug(f"Target speaker repr shape: {target_repr.shape}")

    # Step 4: Multimodal fusion (Conformer encoder + attention) – simplified placeholder
    logger.info("Fusing multimodal features...")
    # In practice: concatenate lyrics embedding, F0, energy, speaker_emb, target_repr and pass through Conformer
    fused_features = torch.cat(
        [speaker_emb, target_repr, f0.unsqueeze(-1), energy.unsqueeze(-1)], dim=-1
    )
    # Placeholder: linear projection to expected vocoder condition dimension
    condition = torch.nn.Linear(fused_features.size(-1), 256)(fused_features)  # dummy

    # Step 5: Generate waveform via neural vocoder
    logger.info("Synthesizing audio with neural vocoder...")
    waveform_gen = vocoder.generate(condition, f0=f0, energy=energy)
    logger.debug(f"Generated waveform shape: {waveform_gen.shape}")

    # Step 6: Quality monitoring & self‑correction loop
    if quality_monitor:
        logger.info("Running quality monitoring...")
        waveform_gen, metrics = quality_monitor.refine(waveform_gen, config['quality_monitor'])
        logger.info(f"Quality metrics: {metrics}")

    # Step 7: Save output (mix vocals with accompaniment if available)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if accompaniment_wav is not None:
        logger.info("Mixing generated vocals with accompaniment...")
        # Ensure vocals are numpy array
        vocal_np = waveform_gen.squeeze().cpu().numpy()
        # Ensure accompaniment is mono numpy array
        if accompaniment_wav.ndim > 1:
            accompaniment_mono = accompaniment_wav.mean(axis=1)
        else:
            accompaniment_mono = accompaniment_wav
        # Resample accompaniment to vocoder output sample rate (24000) if needed
        target_sr = 24000
        if sr != target_sr:
            # Convert to torch tensor for resampling
            acc_torch = torch.from_numpy(accompaniment_mono).float().unsqueeze(0)  # (1, T)
            acc_resampled = F.resample(acc_torch, sr, target_sr)
            accompaniment_np = acc_resampled.squeeze().numpy()
        else:
            accompaniment_np = accompaniment_mono
        # Trim to same length
        min_len = min(len(vocal_np), len(accompaniment_np))
        vocal_np = vocal_np[:min_len]
        accompaniment_np = accompaniment_np[:min_len]
        # Mix
        mixed = vocal_np + accompaniment_np
        # Prevent clipping: simple normalization if peak > 1.0
        peak = np.max(np.abs(mixed))
        if peak > 1.0:
            mixed = mixed / peak
            logger.warning(f"Mixed audio peak exceeded 1.0, normalized by factor {peak:.3f}")
        # Convert to torch tensor and save
        mixed_tensor = torch.from_numpy(mixed).float().unsqueeze(0)
        torchaudio.save(str(output_path), mixed_tensor, sample_rate=target_sr)
        logger.info(f"Cover audio (mixed vocals+accompaniment) saved to: {output_path}")
    else:
        # No accompaniment, save vocals only
        torchaudio.save(str(output_path), waveform_gen.cpu(), sample_rate=24000)
        logger.info(f"Cover audio (vocals only) saved to: {output_path}")

    # Clean up temporary vocals file if we created one
    if vocals_wav is not None and 'vocals_path' in locals() and vocals_path != args.input:
        try:
            os.remove(vocals_path)
            logger.debug(f"Removed temporary vocals file: {vocals_path}")
        except OSError:
            pass


if __name__ == "__main__":
    main()