# Multilingual Speech-to-Speech Translation Pipeline

## Overview
This system provides end-to-end speech-to-speech translation using a cascaded architecture:
1. **ASR (Speech-to-Text):** Faster-Whisper (`small`)
2. **MT (Translation):** Meta NLLB-200 (`facebook/nllb-200-distilled-600M`)
3. **TTS (Text-to-Speech):** Meta MMS (`facebook/mms-tts-{lang}`) using VITS architecture

## Setup & Model Download
You do not need to manually download model files or include weights in this repository.

1. Create and activate a virtual environment:
   ```bash
   python -m venv s2st_env
   source s2st_env/bin/activate  # On Windows: s2st_env\Scripts\activate
