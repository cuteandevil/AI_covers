"""
Basic import test for AI Cover Generator modules.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    try:
        from modules.asr import ASRModule
        from modules.feature_extractor import FeatureExtractor
        from modules.adaptive_learner import AdaptiveLearner
        from modules.neural_vocoder import NeuralVocoder
        from modules.quality_monitor import QualityMonitor
        from utils.logger import setup_logger
        from utils.audio_utils import load_wav, save_wav
        print("All imports succeeded")
        return True
    except Exception as e:
        print(f"Import failed: {e}")
        return False

if __name__ == "__main__":
    test_imports()