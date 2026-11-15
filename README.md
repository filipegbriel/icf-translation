# Church captions: German → English, live

Audio → Silero VAD → faster-whisper (German) → Opus-MT (de→en) → WebSocket → full-screen caption page.
Everything runs locally; the internet is only needed once, to download the models.

## 1. Install (once)

Use Python 3.11 (3.10–3.12 work).

```bash
cd church-captions
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If pip complains about torch or numpy on your Mac, send me the error. Those pins are the usual trouble spot on Intel Macs.

## 2. Test with a recording (no microphone needed)

```bash
python server.py --file test.wav
```

Open http://localhost:8000 in Chrome and press **f** for full screen. The first start downloads the models (a few hundred MB to 1.5 GB) and takes a minute. The recording is played in real time, so you see what a live service would look like. The terminal shows, per segment:

- `asr` – seconds Whisper needed. It must stay well below the `audio` length.
- `delay after speech` – how long the audience waits after the speaker pauses.
- the German text and the English translation.

If you see `WARNING: transcription is falling behind`, use a smaller model (`--model base`) or wait for the GPU machine.

## 3. Live from a microphone or the sound board

```bash
python server.py --list-devices       # find the index of your input
python server.py --input 2            # use that index
```

macOS asks for microphone permission for Terminal the first time. Later, use the USB interface fed from the sound board (speaker mic bus only).

## 4. Keys on the caption page

| Key | Action |
|---|---|
| f | full screen |
| + / - | text size (remembered) |
| c | captions on/off (blanks every connected screen) |
| t | show a test line, to check readability from the back row |

For an OBS browser source, use `http://localhost:8000/overlay` (transparent background, outlined text).

## 5. Useful options

| Option | Default | Meaning |
|---|---|---|
| `--model` | small | `base`, `small`, `medium`, `large-v3-turbo` |
| `--device` | cpu | `cuda` on an NVIDIA GPU |
| `--beam` | 1 | higher = a bit more accurate, slower |
| `--silence-ms` | 600 | pause length that ends a segment; lower = faster captions, more fragments |
| `--max-seg` | 10 | longest segment before a forced cut |
| `--save-audio` | off | keep every segment as wav in `logs/audio/` for review |
| `--host 0.0.0.0` | 127.0.0.1 | let other devices on the network open the page |

## 6. Weekly improvement loop

- `logs/YYYY-MM-DD.jsonl` has the German, English, timings and timestamps for every caption.
- Fix recurring mistakes in `glossary.json`; it is reloaded on every segment, with no restart.
  - `asr_prompt`: words Whisper should expect (names, Bible books, terms).
  - `de_fixes`: corrections applied to the German text before translation.
  - `en_fixes`: corrections applied to the English output.
  - `ignore`: phrases Whisper invents on silence; matching segments are dropped.

## 7. Moving to the GPU PC

Same code. Install the normal CUDA build of PyTorch (no pins), then:

```bash
python server.py --device cuda --model large-v3-turbo
```

## Known limits of this first version

- Captions appear per spoken segment (after a pause or at most every `--max-seg` seconds), not word by word. That is deliberate: German sentences need their verb before they can be translated well.
- A forced cut in the middle of a long sentence can split it awkwardly.
- Scripture quotes are translated like normal speech. Showing the official English verse instead is a good next step.

## Docker

Two images are provided: `Dockerfile` (CPU, any machine) and `Dockerfile.gpu` (NVIDIA GPU, Linux).

```bash
# test on the MacBook: put a recording in samples/test.wav, then
docker compose --profile cpu up --build
# open http://localhost:8000
```

On the final Linux GPU PC (needs Docker plus the NVIDIA Container Toolkit):

```bash
docker compose --profile gpu up -d --build
docker compose --profile gpu logs -f
```

Notes:
- **No microphone on a Mac.** Docker Desktop runs a hidden Linux VM and cannot pass the Mac's audio input through. On the Mac, Docker is for file tests only; for live mic tests run `server.py` natively (section 3).
- On Linux the GPU service gets the sound card through `/dev/snd`. Use a USB interface that the desktop's sound server does not already hold open, and find its index with
  `docker compose --profile gpu run --rm captions-gpu python3 server.py --list-devices`, then start with `CAPTIONS_ARGS="--input N"`.
- Models are cached in the `hf-cache` volume, so rebuilding does not download them again.
- `logs/` and `glossary.json` stay on the host, so the weekly review and glossary edits work as before.
- `restart: unless-stopped` brings the service back after a reboot or crash.
