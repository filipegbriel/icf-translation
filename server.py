#!/usr/bin/env python3
"""Local web server: live German (left) and English translation (right)."""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # must run before any library loads

import argparse
import asyncio
import json
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse

from transcription import Pipeline

ROOT = pathlib.Path(__file__).parent
ARGS = None
LOOP = None
PIPE = None
CLIENTS = set()
STATE = {"paused": False, "status": "stopped", "text": ""}


async def broadcast(message):
    data = json.dumps(message, ensure_ascii=False)
    for websocket in list(CLIENTS):
        try:
            await websocket.send_text(data)
        except Exception:
            CLIENTS.discard(websocket)


def emit(message):
    """Schedule websocket updates safely from worker threads."""
    if LOOP is not None and not LOOP.is_closed():
        asyncio.run_coroutine_threadsafe(broadcast(message), LOOP)


def set_status(status, text=""):
    STATE["status"] = status
    STATE["text"] = text
    emit({"type": "status", "state": status, "text": text})


def publish_german(seg_id, german, final):
    """German text as soon as it is recognized (final=False: a preview that may still change)."""
    if not STATE["paused"]:
        emit({"type": "german", "id": seg_id, "de": german, "final": final})


def publish_caption(seg_id, german, english):
    """English translation for the segment with this id."""
    if not STATE["paused"]:
        emit({"type": "english", "id": seg_id, "en": english})


@asynccontextmanager
async def lifespan(app):
    global LOOP
    LOOP = asyncio.get_running_loop()
    yield
    if PIPE is not None:
        PIPE.shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/overlay")
async def overlay():
    return RedirectResponse("/?overlay=1")      # English only, transparent, for OBS


@app.post("/api/start")
async def start_transcription():
    global PIPE
    if ARGS is None:
        raise HTTPException(status_code=503, detail="Start the app with server.py")
    if PIPE is not None and PIPE.is_running:
        return {"state": STATE["status"]}

    set_status("loading", "Loading Whisper model…")
    PIPE = Pipeline(ARGS, publish_caption, set_status, publish_german)
    PIPE.start()
    return {"state": "loading"}


@app.post("/api/stop")
async def stop_transcription():
    if PIPE is not None and PIPE.is_running:
        set_status("stopping", "Stopping transcription…")
        PIPE.shutdown()
    else:
        set_status("stopped", "Transcription stopped")
    return {"state": STATE["status"]}


@app.post("/api/toggle")
async def toggle():
    STATE["paused"] = not STATE["paused"]
    await broadcast({"type": "pause", "paused": STATE["paused"]})
    return {"paused": STATE["paused"]}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    CLIENTS.add(websocket)
    await websocket.send_text(json.dumps({"type": "pause", "paused": STATE["paused"]}))
    await websocket.send_text(json.dumps({"type": "status", "state": STATE["status"],
                                         "text": STATE["text"]}, ensure_ascii=False))
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        CLIENTS.discard(websocket)


def parse_args():
    parser = argparse.ArgumentParser(description="German speech transcription web server")
    parser.add_argument("--model", default="base", help="Whisper model: tiny, base, small, medium, large-v3-turbo…")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--compute", default=None, help="default: int8 on cpu, float16 on cuda")
    parser.add_argument("--beam", type=int, default=1, help="Whisper beam size (1 = fastest)")
    parser.add_argument("--input", type=int, default=None, help="audio input device index")
    parser.add_argument("--file", default=None, help="transcribe a recording instead of using the mic")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed for --file")
    parser.add_argument("--silence-ms", type=int, default=900, help="pause length that ends a segment")
    parser.add_argument("--max-seg", type=float, default=5.0, help="force a cut after this many seconds")
    parser.add_argument("--partials", action="store_true",
                        help="show German while the speaker is still talking (uses more CPU)")
    parser.add_argument("--partial-s", type=float, default=1.0,
                        help="seconds between German previews (with --partials)")
    parser.add_argument("--save-audio", action="store_true", help="store segments as wav in logs/audio")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other devices")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    if args.compute is None:
        args.compute = "float16" if args.device == "cuda" else "int8"
    return args


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