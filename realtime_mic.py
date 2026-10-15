
import io, queue, threading, time, re
from collections import deque

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from silero_vad import load_silero_vad, VADIterator

from pipeline import MultilingualS2STPipeline, LANG_MAP

# ---------------- change these ----------------
SOURCE_LANG = "en"             # language you speak: en, ml, hi, fr, es
TARGET_NLLB = "mal_Mlym"       # NLLB code of output language
TARGET_TTS = "mal"             # MMS-TTS code of output language
STOP_PHRASES = ["stop translation", "stop listening", "stop the translation"]
# ----------------------------------------------

SR = 16000
BLOCK = 512                                   # Silero needs 512 samples @ 16 kHz
MAX_BLOCKS = int(15 * SR / BLOCK)             # force-cut an utterance at 15 s

audio_q = queue.Queue()                       # raw mic chunks
utt_q = queue.Queue()                         # complete sentences waiting to be translated
speaking = threading.Event()                  # mic muted while playing (prevents echo)
stop_event = threading.Event()                # set => everything shuts down

pipeline = MultilingualS2STPipeline()


def is_stop_command(text: str) -> bool:
    clean = re.sub(r"[^a-z ]", "", text.lower()).strip()
    return any(p in clean for p in STOP_PHRASES)


def keyboard_listener():
    while not stop_event.is_set():
        try:
            if input().strip().lower() == "q":
                print("Stop requested from keyboard.")
                stop_event.set()
        except EOFError:
            return


def mic_callback(indata, frames, t, status):
    if not speaking.is_set() and not stop_event.is_set():
        audio_q.put(indata[:, 0].copy())


def to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def worker():
    while not stop_event.is_set():
        try:
            audio = utt_q.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            t0 = time.time()
            text, _ = pipeline.transcribe(to_wav_bytes(audio), SOURCE_LANG)
            if not text:
                continue
            print(f"\nYou:        {text}")

            if is_stop_command(text):          # check BEFORE translating
                print("Voice stop command heard. Shutting down...")
                stop_event.set()
                break

            translated = pipeline.translate(text, LANG_MAP.get(SOURCE_LANG, "eng_Latn"), TARGET_NLLB)
            print(f"Translated: {translated}")
            audio_bytes, _ = pipeline.synthesize(translated, TARGET_TTS)
            print(f"            (ready in {time.time() - t0:.1f}s)")

            wave, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            speaking.set()
            sd.play(wave, sr)
            while sd.get_stream().active and not stop_event.is_set():
                time.sleep(0.05)
            sd.stop()
        except Exception as e:
            print("Pipeline error:", e)
        finally:
            with audio_q.mutex:
                audio_q.queue.clear()
            speaking.clear()


def main():
    pipeline.warmup(SOURCE_LANG, TARGET_NLLB, TARGET_TTS)
    vad = VADIterator(load_silero_vad(), sampling_rate=SR, threshold=0.5,
                      min_silence_duration_ms=450, speech_pad_ms=200)

    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=keyboard_listener, daemon=True).start()

    preroll, buf, active = deque(maxlen=8), [], False
    print(f"\nLISTENING  ({SOURCE_LANG} -> {TARGET_TTS})")
    print("Speak a sentence, then pause. Say 'stop translation' or type q + Enter to stop.\n")

    try:
        with sd.InputStream(samplerate=SR, channels=1, blocksize=BLOCK,
                            dtype="float32", callback=mic_callback):
            while not stop_event.is_set():
                try:
                    chunk = audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue

                ev = vad(torch.from_numpy(chunk))
                if ev and "start" in ev:
                    active, buf = True, list(preroll)
                if active:
                    buf.append(chunk)
                else:
                    preroll.append(chunk)

                ended = ev and "end" in ev
                if active and (ended or len(buf) > MAX_BLOCKS):
                    active = False
                    if len(buf) * BLOCK / SR > 0.4:        # ignore clicks / noise
                        utt_q.put(np.concatenate(buf))
                    buf = []
                    vad.reset_states()
    except KeyboardInterrupt:
        print("\nCtrl+C pressed.")
    finally:
        stop_event.set()
        sd.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()