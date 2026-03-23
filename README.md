# AI Cover Generator

Industrial-grade system for fully automatic high-quality cover audio generation, addressing shortcomings of existing SVC (Singing Voice Conversion) and RVC (Retrieval-based Voice Conversion) systems.

## Features

- **End-to-end automation**: From raw audio input to final cover output, no manual intervention.
- **Multi-language/multi-dialect support**: Chinese, English, etc.
- **High-quality voice synthesis**: Output close to human singing level, minimal mechanical artifacts.
- **Low latency**: Optimized for online service scenarios (<2s inference).
- **Robustness**: Stable performance under varying recording conditions and background noise.
- **Adaptive few-shot/zero-shot speaker adaptation**: Quickly switch to new voice with ≤3 minutes of adaptation data.
- **Real-time quality monitoring & self-correction**: Detects and fixes artifacts automatically.
- **Edge computing deployment**: Lightweight models deployable on CDN nodes via TensorRT/OpenVINO.
- **Optional source separation**: Built‑in Demucs‑based vocal/accompaniment separation enables using mixed audio as input (enable via `config.yaml`). When enabled, the final output mixes the converted vocals with the accompaniment.
- **Graphical User Interface**: Easy-to-use GUI for generating covers without command-line knowledge.

## Architecture

```
input audio
    │
    ├─► [Optional] Source Separation (Demucs) → vocals
    │                                 (accompaniment saved optionally)
    │
    ├─► ASR (Whisper) → lyric text
    │
    ├─► Audio Frontend → F0, energy, speaker embedding
    │
    └─► Multimodal Fusion (Conformer encoder + attention)
            │
            ▼
    Adaptive Meta‑Learner (MAML‑style) → personalized voice parameters
            │
            ▼
    Neural Vocoder (WaveNet/WaveGlow) → waveform generation
            │
            ▼
    Quality Monitor → artifact detection & self‑correction loop
            │
            ▼
    [Optional] Mix with accompaniment (if separation enabled)
            │
            ▼
    Output cover audio
```

## Getting Started

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repo-url> D:\AI_covers
   cd D:\AI_covers
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pretrained models** (see `scripts/download_models.py` or place them in `models/`):
   - Whisper ASR model (`tiny.en` or `base`)
   - Speaker encoder (e.g., ECAPA-TDNN)
   - F0 extractor (CREPE or Dio)
   - Neural vocoder (WaveGlow or HiFi‑GAN)

5. **Run the demo**:
   ```bash
   python main.py --input demo_input.wav --target_speaker example_speaker --output output_cover.wav
   ```

## Using the GUI

The project also provides a graphical user interface (GUI) for easier interaction.

1. Ensure you have the project dependencies installed (as per `requirements.txt` and the README).
2. Activate the virtual environment if you use one:
   ```bash
   cd D:\AI_covers
   venv\Scripts\activate
   ```
3. Launch the GUI:
   ```bash
   python gui.py
   ```
4. In the GUI window:
   - Select your input audio file (the song you want to convert)
   - Select the target speaker audio (reference voice for the cover)
   - Choose where to save the generated cover audio
   - Configure optional settings (device, accompaniment saving, source separation)
   - Click "Generate Cover" to start the process
   - Monitor progress in the log window

## Configuration

Edit `config.yaml` to adjust:
- Paths to models and data
- ASR language
- Vocoder type
- Latency / quality trade‑offs
- Edge deployment flags

## License

This project is licensed under the MIT License.

## Acknowledgments

- Whisper (OpenAI)
- ECAPA-TDNN speaker embedding
- CREPE F0 estimator
- WaveGlow / HiFi‑GAN vocoders
- MAML for few‑shot adaptation