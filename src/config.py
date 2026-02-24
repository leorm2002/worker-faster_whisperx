# --- Models ---
# Unica lista: i modelli qui vengono pre-scaricati al build e sono gli unici accettati a runtime.
MODELS = ["tiny", "turbo", "large-v2"]

DEFAULT_MODEL = "large-v2"

# --- VAD ---

AVAILABLE_VAD_METHODS = {"pyannote", "silero"}
DEFAULT_VAD_METHOD = "silero"

# --- Compute ---

COMPUTE_TYPE_CPU = "int8"

# --- WhisperX transcribe defaults ---

DEFAULT_BATCH_SIZE = 16

# --- WhisperX asr_options defaults (exposed options only) ---
# Note: Other parameters (temperatures, patience, length_penalty, etc.) 
# rely on the optimal upstream defaults from WhisperX/Faster-Whisper.

DEFAULT_BEAM_SIZE = 5
DEFAULT_LOG_PROB_THRESHOLD = -1.0
DEFAULT_NO_SPEECH_THRESHOLD = 0.6
DEFAULT_CONDITION_ON_PREVIOUS_TEXT = False
DEFAULT_INITIAL_PROMPT = None

# --- Output formatting ---

DEFAULT_TRANSCRIPTION_FORMAT = "plain_text"
DEFAULT_TRANSLATION_FORMAT = "plain_text"