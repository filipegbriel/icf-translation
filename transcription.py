"""Microphone/file audio capture, speech detection, and German transcription."""
import collections
import datetime
import json
import pathlib
import queue
import re
import sys
import threading
import time
import wave

import numpy as np

ROOT = pathlib.Path(__file__).parent
LOGS = ROOT / "logs"
SR = 16000
BLOCK = 512
MIN_SEG_S = 0.5


def load_glossary():
    """Re-read transcription corrections for each segment."""
    glossary = {}
    path = ROOT / "glossary.json"
    if path.exists():
        try:
            glossary = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            print(f"glossary.json error: {error}", file=sys.stderr)
    for key, default in (("asr_prompt", ""), ("de_fixes", {}), ("ignore", [])):
        glossary.setdefault(key, default)
    glossary.setdefault("en_fixes", {})
    return glossary


def apply_fixes(text, fixes):
    for wrong, right in fixes.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE)
    return text


def save_wav(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SR)
        wav_file.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def mic_source(device, stop):
    import sounddevice as sd

    audio_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=BLOCK, device=device, callback=callback):
        print("Listening on the microphone…")
        while not stop.is_set():
            try:
                yield audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue


def file_source(path, speed, stop):
    """Play a recording through the same pipeline for testing."""
    from faster_whisper.audio import decode_audio

    audio = decode_audio(path, sampling_rate=SR)
    step = BLOCK / SR / speed
    next_t = time.time()
    print(f"Playing {path} ({len(audio) / SR:.0f} s) at {speed}x…")
    for index in range(0, len(audio) - BLOCK, BLOCK):
        if stop.is_set():
            return
        yield audio[index:index + BLOCK]
        next_t += step
        time.sleep(max(0.0, next_t - time.time()))
    for _ in range(SR // BLOCK):
        if stop.is_set():
            return
        yield np.zeros(BLOCK, dtype=np.float32)
    print("End of file.")


class Pipeline:
    def __init__(self, args, on_caption, on_status):
        self.args = args
        self.on_caption = on_caption
        self.on_status = on_status
        self.segments = queue.Queue()
        self.stop = threading.Event()
        self.thread = None
        self.asr_thread = None

    @property
    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def shutdown(self):
        self.stop.set()

    def _run(self):
        failed = False
        try:
            self._load_model()
            self.on_status("loading", "Loading German-English translation model…")
            from translation import Translator
            glossary = load_glossary()
            translation_device = "cuda" if self.args.device == "cuda" else "cpu"
            self.translator = Translator(translation_device, glossary["en_fixes"])
            self.asr_thread = threading.Thread(target=self._asr_loop, daemon=True)
            self.asr_thread.start()
            self.on_status("ready", "Listening for German speech")
            self._vad_loop(self._source())
        except Exception as error:
            failed = True
            import traceback
            traceback.print_exc()
            self.on_status("error", f"Error: {error}")
        finally:
            self.stop.set()
            if self.asr_thread is not None:
                self.segments.put(None)
                self.asr_thread.join()
            if not failed:
                self.on_status("stopped", "Transcription stopped")

    def _source(self):
        if self.args.file:
            return file_source(self.args.file, self.args.speed, self.stop)
        return mic_source(self.args.input, self.stop)

    def _load_model(self):
        from faster_whisper import WhisperModel
        import torch
        torch.set_num_threads(1)   # avoids thread clash with CTranslate2 on Intel Macs
        args = self.args
        print(f"Loading Whisper '{args.model}' ({args.device}, {args.compute})…")
        self.asr = WhisperModel(args.model, device=args.device, compute_type=args.compute)

    def _vad_loop(self, source):
        import torch
        from silero_vad import VADIterator, load_silero_vad

        vad = VADIterator(load_silero_vad(), threshold=0.5, sampling_rate=SR,
                          min_silence_duration_ms=self.args.silence_ms, speech_pad_ms=100)
        preroll = collections.deque(maxlen=10)
        buffer, active = [], False
        for block in source:
            if self.stop.is_set():
                break
            event = vad(torch.from_numpy(block), return_seconds=False)
            if event and "start" in event:
                active, buffer = True, list(preroll)
            if active:
                buffer.append(block)
                ended = bool(event and "end" in event)
                too_long = len(buffer) * BLOCK / SR >= self.args.max_seg
                if ended or too_long:
                    self._flush(buffer)
                    buffer, active = [], False
                    if too_long and not ended:
                        vad.reset_states()
            else:
                preroll.append(block)
        if active and buffer:
            self._flush(buffer)

    def _flush(self, blocks):
        audio = np.concatenate(blocks)
        if len(audio) / SR < MIN_SEG_S:
            return
        if self.segments.qsize() >= 3:
            print("WARNING: transcription is falling behind real time. "
                  "Use a smaller model or a GPU.", file=sys.stderr)
        self.segments.put((audio, time.time()))

    def _asr_loop(self):
        while True:
            item = self.segments.get()
            if item is None:
                return
            audio, captured_at = item
            try:
                self._process(audio, captured_at)
            except Exception as error:
                print(f"Processing error: {error}", file=sys.stderr)

    def _process(self, audio, captured_at):
        glossary = load_glossary()
        started_at = time.time()
        segments, _ = self.asr.transcribe(
            audio, language="de", beam_size=self.args.beam, vad_filter=False,
            condition_on_previous_text=False, temperature=0.0)
        segments = [segment for segment in segments if segment.no_speech_prob < 0.9]
        asr_seconds = time.time() - started_at

        german = re.sub(r"\s+", " ", " ".join(s.text.strip() for s in segments)).strip()
        if not german or any(bad.lower() in german.lower() for bad in glossary["ignore"]):
            return
        german = apply_fixes(german, glossary["de_fixes"])

        translation_started = time.time()
        english = self.translator.translate(german)
        translation_seconds = time.time() - translation_started

        now = datetime.datetime.now()
        latency = time.time() - captured_at
        duration = len(audio) / SR
        print(f"[{now:%H:%M:%S}] audio {duration:.1f}s | asr {asr_seconds:.1f}s | "
              f"translation {translation_seconds:.1f}s | delay after speech {latency:.1f}s\n"
              f"  DE: {german}\n  EN: {english}")
        self.on_caption(german, english)

        LOGS.mkdir(exist_ok=True)
        with open(LOGS / f"{now:%Y-%m-%d}.jsonl", "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps({
                "ts": now.isoformat(timespec="seconds"), "de": german, "en": english,
                "audio_s": round(duration, 2), "asr_s": round(asr_seconds, 2),
                "translation_s": round(translation_seconds, 2),
                "delay_s": round(latency, 2)}, ensure_ascii=False) + "\n")
        if self.args.save_audio:
            save_wav(LOGS / "audio" / f"{now:%Y-%m-%d_%H-%M-%S}.wav", audio)