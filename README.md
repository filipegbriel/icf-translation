# Live German-to-English captions

The app has three parts: `server.py` serves the control/caption page, `transcription.py` handles audio input, voice detection, and German speech recognition, and `translation.py` translates each German segment into English. Start the server in a terminal, open the page, then click **Start listening**. The page shows the German transcript above the English translation. The web server stays running when you click **Stop**.

Speech recognition and translation run locally. Internet access is needed the first time to download the Whisper and German-English translation models.

## Install for local use

Use Python 3.11 (3.10–3.12 work). From the project directory:

```bash
cd icf-translation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Debian/Ubuntu, install PortAudio for microphone access:

```bash
sudo apt install libportaudio2
```

## Test the provided recording

Make sure `samples/test.wav` exists, then start the server with the recording as its input:

```bash
cd /home/kaique/Documents/git/icf-translation
python server.py --host 127.0.0.1 --port 8000 --file samples/test.wav
```

Open http://127.0.0.1:8000 and click **Start listening**. The first start downloads/loads both models and can take a few minutes. The recording plays at its normal speed through the same recognition and translation pipeline as a microphone. When it ends, the page returns to **Stopped**. To stop the server itself, press `Ctrl+C` in its terminal.

## Use a microphone

List the audio devices, then start the web server with the input device index. Keep this terminal open:

```bash
python server.py --list-devices
python server.py --host 127.0.0.1 --port 8000 --input 2
```

Replace `2` with the index for your microphone or sound-board interface. Open http://127.0.0.1:8000 and click **Start listening**. The server process captures audio directly from its host. On macOS, grant microphone permission to Terminal. Docker Desktop on macOS cannot pass the Mac microphone through to a Linux container; run the server natively for Mac microphone use.

## Page controls

| Control | Action |
|---|---|
| Start listening | Start audio capture, German transcription, and English translation |
| Stop | Stop audio processing without stopping the web server |
| `f` | Toggle full screen |
| `+` / `-` | Increase or decrease text size; the setting is remembered |
| `c` | Hide or show captions |
| `t` | Show a sample line to check readability |

For an OBS browser source, use http://127.0.0.1:8000/overlay. It hides the operator controls and uses a transparent background with outlined text.

## Performance options

| Option | Default | Meaning |
|---|---|---|
| `--model` | `base` | Whisper model. `tiny` is faster; `small` or larger may improve recognition but need more processing time. |
| `--device` | `cpu` | Use `cuda` on a supported NVIDIA GPU. |
| `--beam` | `1` | Higher values may improve accuracy but run slower. |
| `--silence-ms` | `400` | Silence duration that ends a speech segment. Higher values wait longer and can join phrases. |
| `--max-seg` | `5` | Force a segment boundary after this many seconds. |
| `--save-audio` | off | Save each recognized audio segment under `logs/audio/`. |
| `--host` | `127.0.0.1` | Use `0.0.0.0` to allow other devices on the network to open the page. |
| `--port` | `8000` | Web server port. |

Shorter segments can reduce the time before a caption appears, but may split a sentence and affect recognition or translation. Translation adds processing time after Whisper. If the warning `transcription is falling behind real time` appears, try `--model tiny` or run on a GPU. The first model load/download also takes extra time.

Captions are emitted by speech segment, not word by word. Translation is generated one segment at a time; it may be less natural when a segment ends in the middle of a sentence. This step does not speak the English output yet. JSONL German/English text and timing data are written to `logs/YYYY-MM-DD.jsonl`. `glossary.json` supports `de_fixes` for recognized-text corrections, `en_fixes` for translation corrections, and `ignore` for phrases to drop; it is re-read for each segment.

## Docker CPU test

The CPU Compose profile uses `samples/test.wav` by default. Stop any local server using port 8000 first, then run:

```bash
cd /home/kaique/Documents/git/icf-translation
docker compose --profile cpu up --build
```

Open http://127.0.0.1:8000 and click **Start listening**. The recording is replayed each time processing starts. Press `Ctrl+C` in the Compose terminal to stop the container, or use `docker compose --profile cpu down` from another terminal. Both models are cached in the mounted `hf-cache` volume.

If Docker reports permission denied for `/var/run/docker.sock`, your account cannot access the Docker daemon. On Ubuntu, an administrator can add your account to the `docker` group with `sudo usermod -aG docker "$USER"`; log out and back in for the new group to apply. Membership in that group grants root-equivalent privileges.

## Docker with an NVIDIA GPU

The GPU profile is intended for a Linux host with Docker Compose, the NVIDIA Container Toolkit, and an NVIDIA GPU. It uses the microphone or sound interface exposed to the container through `/dev/snd`:

```bash
docker compose --profile gpu up --build
```

To list devices inside the container, use:

```bash
docker compose --profile gpu run --rm captions-gpu python3 server.py --list-devices
```

Then pass the selected index on startup, for example:

```bash
CAPTIONS_ARGS="--input 2" docker compose --profile gpu up --build
```

Use `CAPTIONS_ARGS="--file /samples/test.wav"` for a file test instead. Models are cached in the `hf-cache` volume; logs and the glossary remain in the project directory.

## Port already in use

If the server reports `Address already in use`, find the process using port 8000:

```bash
ss -ltnp 'sport = :8000'
```

Stop that server with `Ctrl+C` in its terminal. If its terminal is unavailable, send `kill -INT PID`, replacing `PID` with the process ID reported by the command. Alternatively, start this server on another port with `--port 8001` and open http://127.0.0.1:8001.
