"""
ASR module using Whisper for lyric transcription.
"""
import torch
import whisper
import logging
from typing import Union

logger = logging.getLogger(__name__)

class ASRModule:
    def __init__(self, config: dict, device: str = "cpu"):
        self.device = device
        model_name = config.get("model_name", "openai/whisper-base.en")
        # whisper.load_model expects model size or path
        # map huggingface-like names to whisper sizes
        size_map = {
            "openai/whisper-tiny.en": "tiny",
            "openai/whisper-base.en": "base",
            "openai/whisper-small.en": "small",
            "openai/whisper-medium.en": "medium",
            "openai/whisper-large-v2": "large",
        }
        model_size = size_map.get(model_name, model_name.split("/")[-1].split('.')[0])
        logger.info(f"Loading Whisper model '{model_size}' on {device}")
        self.model = whisper.load_model(model_size, device=device)
        self.language = config.get("language", "en")
        self.task = config.get("task", "transcribe")

    def transcribe(self, audio_path: Union[str, torch.Tensor]) -> str:
        """
        Transcribe audio to text (lyrics).
        Returns the transcribed string.
        """
        logger.debug(f"Transcribing {audio_path}")
        result = self.model.transcribe(
            audio_path,
            language=self.language,
            task=self.task
        )
        text = result["text"].strip()
        logger.debug(f"Transcription: {text}")
        return text