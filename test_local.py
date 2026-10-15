import soundfile as sf
import numpy as np
from pipeline import MultilingualS2STPipeline

# 1. Initialize pipeline
pipeline = MultilingualS2STPipeline()

# 2. Open your input audio file (e.g., harvard.wav)
try:
    with open("harvard.wav", "rb") as f:
        audio_data = f.read()
except FileNotFoundError:
    print("harvard.wav not found! Please place a wav file in the folder.")
    exit()

# 3. Run translation for Malayalam
# NLLB code for Malayalam: 'mal_Mlym', MMS TTS code: 'mal'
result = pipeline.run_pipeline(
    audio_bytes=audio_data, 
    target_nllb = "eng_Latn"
    target_tts = "eng"
)

# 4. Save the generated Malayalam audio
with open("translated_output_malayalam.wav", "wb") as f:
    f.write(result["audio_bytes"])

print("Translation complete! Saved output to translated_output_malayalam.wav")