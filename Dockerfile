FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
# Sopprime i warning di ONNXRuntime sulla GPU discovery (DRM non accessibile in container)
ENV ORT_LOG_SEVERITY_LEVEL=3
ENV TORCH_HOME=/models/torch
WORKDIR /

# 1. Installazioni di sistema (Cambiano quasi mai)
RUN apt-get update -y && \
    apt-get install --yes --no-install-recommends \
        sudo ca-certificates git wget curl bash \
        libgl1 libx11-6 software-properties-common ffmpeg build-essential && \
    apt-get clean -y && rm -rf /var/lib/apt/lists/*

# 2. Pip Upgrade e Requirements (Cambiano solo se aggiungi librerie)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r /requirements.txt && \
    # torchcodec non è compatibile con PyTorch 2.8.0 e manca libnppicc.so.12;
    # pyannote.audio ha un fallback funzionante, quindi lo rimuoviamo
    pip uninstall -y torchcodec || true && \
    # Aggiorna il checkpoint pyannote da Lightning v1.5.4 a v2.6.1 in modo permanente
    python3 -c "import whisperx, os; p = os.path.join(os.path.dirname(whisperx.__file__), 'assets', 'pytorch_model.bin'); os.system(f'python3 -m lightning.pytorch.utilities.upgrade_checkpoint {p}')"

# 3. Download dei Modelli (Pesanti, da fare UNA volta sola)
# config.py serve a fetch_models.py per leggere MODELS e COMPUTE_TYPE_CPU
COPY src/config.py /config.py
COPY builder/fetch_models.py /fetch_models.py
RUN python3 /fetch_models.py && rm /fetch_models.py /config.py

# 4. Codice sorgente (Cambia SEMPRE)
# Copiamo il codice solo alla fine.
COPY src .
COPY test_input.json .

CMD ["python3", "-u", "/rp_handler.py"]