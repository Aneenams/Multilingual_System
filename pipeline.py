"""
Faster drop-in replacement for pipeline.py.
Same class name and same run_pipeline() signature, so test_mic.py,
server.py and realtime_mic.py keep working unchanged.

Needs:  pip install faster-whisper
"""
import io
import time
import torch
import torchaudio
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoProcessor, VitsModel

# ---------------- tweak these ----------------
WHISPER_SIZE = "small"                              # tiny / base / small  (small = good accuracy)
NLLB_MODEL = "facebook/nllb-200-distilled-600M"     # ~2x faster than 1.3B
NUM_BEAMS = 2                                       # 1 = fastest, 5 = slowest/most accurate
QUANTIZE_NLLB_ON_CPU = True                         # int8 on CPU: faster, tiny quality loss
# ---------------------------------------------

LANG_MAP = {
    "en": "eng_Latn",
    "ml": "mal_Mlym",
    "hi": "hin_Deva",
    "fr": "fra_Latn",
    "es": "spa_Latn",
}


class MultilingualS2STPipeline:
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            torch.set_num_threads(torch.get_num_threads())
        print(f"Loading models on device: {self.device}")

        print(f"Loading ASR (faster-whisper {WHISPER_SIZE})...")
        compute = "float16" if self.device == "cuda" else "int8"
        self.asr = WhisperModel(WHISPER_SIZE, device=self.device, compute_type=compute)

        print(f"Loading Translation Model ({NLLB_MODEL})...")
        self.translator_tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
        if self.device == "cpu" and QUANTIZE_NLLB_ON_CPU:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        self.translator_model = model.to(self.device).eval()

        self.tts_cache = {}

    # ---------- TTS loader ----------
    def _get_tts_model(self, lang_code: str):
        if lang_code not in self.tts_cache:
            model_id = f"facebook/mms-tts-{lang_code}"
            print(f"Loading TTS model for: {lang_code}...")
            processor = AutoProcessor.from_pretrained(model_id)
            model = VitsModel.from_pretrained(model_id).to(self.device).eval()
            self.tts_cache[lang_code] = (processor, model)
        return self.tts_cache[lang_code]

    # ---------- warm-up (call once at startup) ----------
    def warmup(self, source_lang="en", target_nllb_lang="hin_Deva", tts_lang_code="hin"):
        """Runs every model once so the first real request isn't slow."""
        print("Warming up models...")
        t0 = time.time()
        list(self.asr.transcribe(np.zeros(16000, dtype=np.float32), language=source_lang)[0])
        text = self.translate("Hello, how are you?", LANG_MAP.get(source_lang, "eng_Latn"), target_nllb_lang)
        self.synthesize(text, tts_lang_code)
        print(f"Warm-up done in {time.time() - t0:.1f}s")

    # ---------- ASR ----------
    def transcribe(self, audio_bytes: bytes, source_lang: str = "en") -> tuple[str, str]:
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sample_rate != 16000:
            tensor = torch.from_numpy(audio).unsqueeze(0)
            tensor = torchaudio.transforms.Resample(sample_rate, 16000)(tensor).squeeze(0)
            audio = tensor.numpy()

        segments, _ = self.asr.transcribe(
            audio,
            language=source_lang,
            beam_size=1,                       # greedy = fastest
            condition_on_previous_text=False,
            vad_filter=False,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        return text, source_lang

    # ---------- Translation ----------
    def translate(self, text: str, source_nllb_lang: str, target_nllb_lang: str) -> str:
        self.translator_tokenizer.src_lang = source_nllb_lang
        inputs = self.translator_tokenizer(text, return_tensors="pt").to(self.device)
        forced_bos = self.translator_tokenizer.convert_tokens_to_ids(target_nllb_lang)

        with torch.inference_mode():
            tokens = self.translator_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=128,
                num_beams=NUM_BEAMS,
            )
        return self.translator_tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    # ---------- TTS ----------
    def synthesize(self, text: str, tts_lang_code: str) -> tuple[bytes, int]:
        processor, model = self._get_tts_model(tts_lang_code)
        inputs = processor(text=text, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            waveform = model(**inputs).waveform

        waveform = waveform.squeeze().cpu().numpy()
        sample_rate = model.config.sampling_rate

        buf = io.BytesIO()
        sf.write(buf, waveform, sample_rate, format="WAV")
        return buf.getvalue(), sample_rate

    # ---------- full pipeline ----------
    def run_pipeline(self, audio_bytes: bytes, target_nllb_lang: str, tts_lang_code: str, source_lang: str = "en"):
        t0 = time.time()
        transcript, detected_lang = self.transcribe(audio_bytes, source_lang=source_lang)
        t1 = time.time()
        print(f"[ASR  {t1 - t0:.2f}s] {transcript}")

        if not transcript:
            return {"transcript": "", "detected_language": detected_lang, "translation": "",
                    "audio_bytes": b"", "sample_rate": 0, "timings": {}}

        source_nllb_lang = LANG_MAP.get(source_lang, "eng_Latn")
        translated_text = self.translate(transcript, source_nllb_lang, target_nllb_lang)
        t2 = time.time()
        print(f"[MT   {t2 - t1:.2f}s] {translated_text}")

        output_audio_bytes, sample_rate = self.synthesize(translated_text, tts_lang_code)
        t3 = time.time()
        print(f"[TTS  {t3 - t2:.2f}s]   TOTAL {t3 - t0:.2f}s")

        return {
            "transcript": transcript,
            "detected_language": detected_lang,
            "translation": translated_text,
            "audio_bytes": output_audio_bytes,
            "sample_rate": sample_rate,
            "timings": {"asr": t1 - t0, "mt": t2 - t1, "tts": t3 - t2, "total": t3 - t0},
        }