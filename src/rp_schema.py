from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_BEAM_SIZE,
    DEFAULT_CONDITION_ON_PREVIOUS_TEXT,
    DEFAULT_INITIAL_PROMPT,
    DEFAULT_LOG_PROB_THRESHOLD,
    DEFAULT_MODEL,
    DEFAULT_NO_SPEECH_THRESHOLD,
    DEFAULT_TRANSCRIPTION_FORMAT,
    DEFAULT_TRANSLATION_FORMAT,
    DEFAULT_VAD_METHOD,
)

INPUT_VALIDATIONS = {
    'audio': {
        'type': str,
        'required': False,
        'default': None
    },
    'audio_base64': {
        'type': str,
        'required': False,
        'default': None
    },
    'model': {
        'type': str,
        'required': False,
        'default': DEFAULT_MODEL
    },
    'transcription': {
        'type': str,
        'required': False,
        'default': DEFAULT_TRANSCRIPTION_FORMAT
    },
    'translate': {
        'type': bool,
        'required': False,
        'default': False
    },
    'translation': {
        'type': str,
        'required': False,
        'default': DEFAULT_TRANSLATION_FORMAT
    },
    'language': {
        'type': str,
        'required': False,
        'default': None
    },
    'beam_size': {
        'type': int,
        'required': False,
        'default': DEFAULT_BEAM_SIZE
    },
    'initial_prompt': {
        'type': str,
        'required': False,
        'default': DEFAULT_INITIAL_PROMPT
    },
    'condition_on_previous_text': {
        'type': bool,
        'required': False,
        'default': DEFAULT_CONDITION_ON_PREVIOUS_TEXT
    },
    'logprob_threshold': {
        'type': float,
        'required': False,
        'default': DEFAULT_LOG_PROB_THRESHOLD
    },
    'no_speech_threshold': {
        'type': float,
        'required': False,
        'default': DEFAULT_NO_SPEECH_THRESHOLD
    },
    'suppress_numerals': {
        'type': bool,
        'required': False,
        'default': False
    },
    'hotwords': {
        'type': str,
        'required': False,
        'default': None
    },
    'word_timestamps': {
        'type': bool,
        'required': False,
        'default': False
    },
    'batch_size': {
        'type': int,
        'required': False,
        'default': DEFAULT_BATCH_SIZE
    },
    'vad_method': {
        'type': str,
        'required': False,
        'default': DEFAULT_VAD_METHOD
    },
}