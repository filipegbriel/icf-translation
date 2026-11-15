#!/usr/bin/env python3
"""Live German -> English captions.

Audio (mic or test file) -> Silero VAD -> faster-whisper (German) -> Opus-MT (de->en)
-> WebSocket -> full-screen caption page (http://localhost:8000).
"""
import argparse
import asyncio
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
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse

ROOT = pathlib.Path(__file__).parent
LOGS = ROOT / "logs"
SR = 16000
BLOCK = 512          # 32 ms; the block size Silero VAD expects at 16 kHz
MIN_SEG_S = 0.5      # ignore snippets shorter than this

ARGS = None
LOOP = None
CLIENTS = set()
STATE = {"paused": False, "status": "loading", "text": "Loading models…"}


# --------------------------------------------------------------------------- helpers
def load_glossary():
    """Re-read on every segment, so you can edit glossary.json while running."""
    g = {}
    path = ROOT / "glossary.json"
    if path.exists():
        try:
            g = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # keep running with a broken file
            print(f"glossary.json error: {e}", file=sys.stderr)
    for key, default in (("asr_prompt", ""), ("de_fixes", {}), ("en_fixes", {}), ("ignore", [])):
        g.setdefault(key, default)
    return g


def apply_fixes(text, fixes):
    for wrong, right in fixes.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text, flags=re.IGNORECASE)
    return text


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def save_wav(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


async def broadcast(msg):
    data = json.dumps(msg, ensure_ascii=False)
    for ws in list(CLIENTS):
        try:
            await ws.send_text(data)
        except Exception:
            CLIENTS.discard(ws)


def emit(msg):
    """Thread-safe: called from the pipeline threads."""
    if LOOP is not None and not LOOP.is_closed():
        asyncio.run_coroutine_threadsafe(broadcast(msg), LOOP)


def set_status(status, text=""):
    STATE["status"] = status
    STATE["text"] = text
    emit({"type": "status", "state": status, "text": text})


# --------------------------------------------------------------------------- audio sources
def mic_source(device, stop):
    import sounddevice as sd

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=BLOCK, device=device, callback=callback):
        print("Listening on the microphone…")
        while not stop.is_set():
            try:
                yield q.get(timeout=0.5)
            except queue.Empty:
                continue


def file_source(path, speed, stop):
    """Plays a recording through the pipeline at real-time speed (for testing)."""
    from faster_whisper.audio import decode_audio

    audio = decode_audio(path, sampling_rate=SR)
    step = BLOCK / SR / speed
    next_t = time.time()
    print(f"Playing {path} ({len(audio) / SR:.0f} s) at {speed}x…")
    for i in range(0, len(audio) - BLOCK, BLOCK):
        if stop.is_set():
            return
        yield audio[i:i + BLOCK]
        next_t += step
        time.sleep(max(0.0, next_t - time.time()))
    for _ in range(SR // BLOCK):  # one second of silence closes the last segment
        yield np.zeros(BLOCK, dtype=np.float32)
    print("End of file.")


# --------------------------------------------------------------------------- pipeline
class Pipeline:
    def __init__(self, args):
        self.args = args
        self.segments = queue.Queue()
        self.stop = threading.Event()
        self.tdev = "cuda" if args.device == "cuda" else "cpu"

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def shutdown(self):
        self.stop.set()

    def _run(self):
        try:
            self._load_models()
            threading.Thread(target=self._asr_loop, daemon=True).start()
            set_status("ready")
            print("Ready. Open http://localhost:%d" % self.args.port)
            self._vad_loop(self._source())
        except Exception as e:
            import traceback
            traceback.print_exc()
            set_status("error", f"Error: {e}")

    def _source(self):
        if self.args.file:
            return file_source(self.args.file, self.args.speed, self.stop)
        return mic_source(self.args.input, self.stop)

    def _load_models(self):
        from faster_whisper import WhisperModel
        from transformers import MarianMTModel, MarianTokenizer

        a = self.args
        print(f"Loading Whisper '{a.model}' ({a.device}, {a.compute})…")
        self.asr = WhisperModel(a.model, device=a.device, compute_type=a.compute)
        print("Loading translation model…")
        name = "Helsinki-NLP/opus-mt-de-en"
        self.tok = MarianTokenizer.from_pretrained(name)
        self.mt = MarianMTModel.from_pretrained(name).to(self.tdev).eval()
        # warm-up so the first real sentence is not slow
        self._translate(["Guten Morgen."])

    def _translate(self, sentences):
        import torch

        batch = self.tok(sentences, return_tensors="pt", padding=True,
                         truncation=True, max_length=256).to(self.tdev)
        with torch.inference_mode():
            out = self.mt.generate(**batch, num_beams=4, max_new_tokens=256)
        return self.tok.batch_decode(out, skip_special_tokens=True)

    # ---- stage 1: voice activity detection -> speech segments
    def _vad_loop(self, source):
        import torch
        from silero_vad import VADIterator, load_silero_vad

        vad = VADIterator(load_silero_vad(), threshold=0.5, sampling_rate=SR,
                          min_silence_duration_ms=self.args.silence_ms, speech_pad_ms=100)
        preroll = collections.deque(maxlen=10)   # ~320 ms before speech starts
        buf, active = [], False
        for block in source:
            ev = vad(torch.from_numpy(block), return_seconds=False)
            if ev and "start" in ev:
                active, buf = True, list(preroll)
            if active:
                buf.append(block)
                ended = bool(ev and "end" in ev)
                too_long = len(buf) * BLOCK / SR >= self.args.max_seg
                if ended or too_long:
                    self._flush(buf)
                    buf, active = [], False
                    if too_long and not ended:
                        vad.reset_states()   # re-trigger on continuing speech
            else:
                preroll.append(block)

    def _flush(self, blocks):
        audio = np.concatenate(blocks)
        if len(audio) / SR < MIN_SEG_S:
            return
        if self.segments.qsize() >= 3:
            print("WARNING: transcription is falling behind real time. "
                  "Use a smaller model or a GPU.", file=sys.stderr)
        self.segments.put((audio, time.time()))

    # ---- stage 2: German ASR -> English MT -> display
    def _asr_loop(self):
        while not self.stop.is_set():
            try:
                audio, t_captured = self.segments.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._process(audio, t_captured)
            except Exception as e:
                print(f"Processing error: {e}", file=sys.stderr)

    def _process(self, audio, t_captured):
        g = load_glossary()
        a = self.args

        t = time.time()
        segs, _ = self.asr.transcribe(
            audio, language="de", beam_size=a.beam, vad_filter=False,
            condition_on_previous_text=False, temperature=0.0,
            initial_prompt=g["asr_prompt"] or None)
        segs = [s for s in segs if s.no_speech_prob < 0.7]
        asr_s = time.time() - t

        de = re.sub(r"\s+", " ", " ".join(s.text.strip() for s in segs)).strip()
        if not de or any(bad.lower() in de.lower() for bad in g["ignore"]):
            return
        de = apply_fixes(de, g["de_fixes"])

        t = time.time()
        en = " ".join(self._translate(split_sentences(de)))
        en = apply_fixes(en, g["en_fixes"])
        mt_s = time.time() - t

        now = datetime.datetime.now()
        latency = time.time() - t_captured
        dur = len(audio) / SR
        print(f"[{now:%H:%M:%S}] audio {dur:.1f}s | asr {asr_s:.1f}s | mt {mt_s:.1f}s | "
              f"delay after speech {latency:.1f}s\n  DE: {de}\n  EN: {en}")

        if not STATE["paused"]:
            emit({"type": "caption", "de": de, "en": en})

        LOGS.mkdir(exist_ok=True)
        with open(LOGS / f"{now:%Y-%m-%d}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": now.isoformat(timespec="seconds"), "de": de, "en": en,
                                "audio_s": round(dur, 2), "asr_s": round(asr_s, 2),
                                "mt_s": round(mt_s, 2), "delay_s": round(latency, 2)},
                               ensure_ascii=False) + "\n")
        if a.save_audio:
            save_wav(LOGS / "audio" / f"{now:%Y-%m-%d_%H-%M-%S}.wav", audio)


# --------------------------------------------------------------------------- web app
PIPE = None


@asynccontextmanager
async def lifespan(app):
    global LOOP, PIPE
    LOOP = asyncio.get_running_loop()
    PIPE = Pipeline(ARGS)
    PIPE.start()
    yield
    PIPE.shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/overlay")
async def overlay():
    return RedirectResponse("/?overlay=1")


@app.post("/api/toggle")
async def toggle():
    STATE["paused"] = not STATE["paused"]
    await broadcast({"type": "pause", "paused": STATE["paused"]})
    return {"paused": STATE["paused"]}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    CLIENTS.add(ws)
    await ws.send_text(json.dumps({"type": "pause", "paused": STATE["paused"]}))
    await ws.send_text(json.dumps({"type": "status", "state": STATE["status"],
                                   "text": STATE["text"]}, ensure_ascii=False))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        CLIENTS.discard(ws)


# --------------------------------------------------------------------------- entry point
def parse_args():
    p = argparse.ArgumentParser(description="Live German -> English captions")
    p.add_argument("--model", default="small", help="Whisper model: small, medium, large-v3-turbo…")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--compute", default=None, help="default: int8 on cpu, float16 on cuda")
    p.add_argument("--beam", type=int, default=1, help="Whisper beam size (1 = fastest)")
    p.add_argument("--input", type=int, default=None, help="audio input device index")
    p.add_argument("--file", default=None, help="play a recording instead of using the mic")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed for --file")
    p.add_argument("--silence-ms", type=int, default=600, help="pause length that ends a segment")
    p.add_argument("--max-seg", type=float, default=10.0, help="force a cut after this many seconds")
    p.add_argument("--save-audio", action="store_true", help="store each segment as wav in logs/audio")
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other devices")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--list-devices", action="store_true")
    a = p.parse_args()
    if a.compute is None:
        a.compute = "float16" if a.device == "cuda" else "int8"
    return a


def main():
    global ARGS
    ARGS = parse_args()
    if ARGS.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    import uvicorn
    uvicorn.run(app, host=ARGS.host, port=ARGS.port, log_level="warning")


if __name__ == "__main__":
    main()
