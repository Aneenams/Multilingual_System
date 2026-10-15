import sounddevice as sd
import soundfile as sf
import io
from pipeline import MultilingualS2STPipeline

# 1. Initialize Pipeline
pipeline = MultilingualS2STPipeline()

duration = 5  # Recording duration in seconds
sample_rate = 16000

print(f"\n--- RECORDING ACTIVE: Speak clearly into your mic for {duration} seconds... ---")
audio_np = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()  # Wait until recording finishes
print("--- RECORDING COMPLETE. Processing automatic language detection & translation ---")

# Convert numpy audio array into in-memory WAV bytes
audio_buffer = io.BytesIO()
sf.write(audio_buffer, audio_np, sample_rate, format='WAV', subtype='PCM_16')
audio_bytes = audio_buffer.getvalue()

# ==========================================
# DEFINE YOUR TARGET TRANSLATION GOAL HERE:
# ==========================================
# Example: Translate whatever you speak into Malayalam
target_nllb_lang = "mal_Mlym"  # NLLB code for Malayalam
tts_lang_code = "mal"          # MMS code for Malayalam

# (If you wanted to translate to Hindi instead, use: "hin_Deva" and "hin")
# (If you wanted to translate to English instead, use: "eng_Latn" and "eng")

# Run through the pipeline
# Run through the pipeline (e.g., speaking in English ('en'), translating to Malayalam)
result = pipeline.run_pipeline(
    audio_bytes=audio_bytes, 
    target_nllb_lang="mal_Mlym", 
    tts_lang_code="mal",
    source_lang="en"  # Change to "ml" if you speak Malayalam into your mic
)
print(f"\n--- RESULTS ---")
print(f"Detected Spoken Language: {result['detected_language']}")
print(f"You Said (Transcript):   {result['transcript']}")
print(f"Translated Text:         {result['translation']}")

# Save the live translated audio output
output_filename = "live_translated_output.wav"
with open(output_filename, "wb") as f:
    f.write(result["audio_bytes"])

print(f"\nSaved translated speech to '{output_filename}'. Play it to check accuracy!")