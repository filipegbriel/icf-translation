# CPU image: works on any machine, including Docker Desktop on a Mac.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends libportaudio2 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY server.py transcription.py glossary.json ./
COPY static ./static

EXPOSE 8000
CMD ["python", "server.py", "--host", "0.0.0.0"]
