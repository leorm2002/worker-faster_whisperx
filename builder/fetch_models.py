import sys
import os
import torch  # <--- Aggiunto
import whisperx
from config import MODELS, COMPUTE_TYPE_CPU

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "/")

# --- WHISPERX ---
for model_name in MODELS:
    print(f"Downloading Whisper model: {model_name}...")
    model = whisperx.load_model(model_name, device="cpu", compute_type=COMPUTE_TYPE_CPU)
    del model
    print(f"Finished downloading {model_name}.")

# --- SILERO VAD ---
print("Downloading Silero VAD...")
# Carichiamo il modello su CPU per forzare il download dei file nella TORCH_HOME
vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      onnx=False,
                                      trust_repo=True)
del vad_model
print("Finished downloading Silero VAD.")

print("All models downloaded.")