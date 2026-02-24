import gc
import hashlib
import json

import torch
import whisperx
from runpod.serverless.utils import rp_cuda

from config import (
    MODELS,
    AVAILABLE_VAD_METHODS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODEL,
    DEFAULT_TRANSCRIPTION_FORMAT,
    DEFAULT_TRANSLATION_FORMAT,
    DEFAULT_VAD_METHOD,
)

class Predictor:
    def __init__(self):
        self._model = None
        self._model_name = None
        self._model_key = None

    def setup(self):
        # Qui potresti opzionalmente pre-caricare il modello di default
        # all'avvio a freddo del worker.
        pass

    def _device(self):
        return "cuda" if rp_cuda.is_available() else "cpu"

    def _load_model(self, model_name, asr_options, vad_method):
        # Generiamo l'hash basandoci solo sulle opzioni effettivamente passate
        options_hash = hashlib.md5(
            json.dumps(asr_options, sort_keys=True, default=str).encode()
        ).hexdigest()
        
        model_key = f"{model_name}:{vad_method}:{options_hash}"

        # Se il modello con queste identiche configurazioni è già in RAM/VRAM, usciamo
        if self._model_key == model_key:
            return

        device = self._device()

        # Pulizia rigorosa del modello precedente
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        print(f"Loading model: {model_name} (vad={vad_method})...")
        self._model = whisperx.load_model(
            model_name,
            device=device,
            asr_options=asr_options,
            vad_method=vad_method,
        )

        self._model_name = model_name
        self._model_key = model_key
        print(f"Model {model_name} loaded successfully.")

    def predict(
        self,
        audio,
        model_name=DEFAULT_MODEL,
        transcription_mode=DEFAULT_TRANSCRIPTION_FORMAT,
        do_translate=False,  # Rinomato per evitare conflitto con il metodo self.translate
        translation_format=DEFAULT_TRANSLATION_FORMAT,
        language=None,
        beam_size=5,
        initial_prompt=None,
        condition_on_previous_text=False,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        suppress_numerals=False,
        hotwords=None,
        enable_word_timestamps=False,
        batch_size=DEFAULT_BATCH_SIZE,
        vad_method=DEFAULT_VAD_METHOD,
    ):
        if model_name not in MODELS:
            raise ValueError(f"Invalid model: {model_name}. Available: {MODELS}")

        if vad_method not in AVAILABLE_VAD_METHODS:
            raise ValueError(f"Invalid vad_method: {vad_method}. Available: {AVAILABLE_VAD_METHODS}")

        # --- ASR OPTIONS (Solo i parametri sicuri ed esposti) ---
        # I parametri ometti (temperature, patience, ecc.) prenderanno 
        # automaticamente i default ideali di Faster-Whisper/WhisperX.
        asr_options = {
            "beam_size": beam_size,
            "condition_on_previous_text": condition_on_previous_text,
            "log_prob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "suppress_numerals": suppress_numerals,
        }
        
        # Aggiungiamo chiavi opzionali solo se l'utente le ha valorizzate
        if initial_prompt:
            asr_options["initial_prompt"] = initial_prompt
        if hotwords:
            asr_options["hotwords"] = hotwords

        # Caricamento dinamico
        self._load_model(model_name, asr_options, vad_method)
        device = self._device()

        # Caricamento audio in array numpy
        audio_array = whisperx.load_audio(str(audio))
        
        # 1. Trascrizione ed eventuale allineamento
        detected_language, transcription, serialized_segments, word_timestamps = self.transcribe(
            transcription_mode, language, enable_word_timestamps, batch_size, device, audio_array
        ) 

        # 2. Eventuale traduzione (Whisper traduce sempre e solo verso l'Inglese)
        translation_output = None
        if do_translate:
            translation_output = self.translate(
                translation_format, detected_language, batch_size, audio_array
            )

        return {
            "segments": serialized_segments,
            "detected_language": detected_language,
            "transcription": transcription,
            "translation": translation_output,
            "word_timestamps": word_timestamps,
            "device": device,
            "model": model_name,
        }

    def transcribe(self, transcription_mode, language, enable_word_timestamps, batch_size, device, audio_array):
        # Trascrizione Base
        result = self._model.transcribe(
            audio_array,
            batch_size=batch_size,
            num_workers=0,
            language=language,
            task="transcribe",
            chunk_size=30,
        )

        detected_language = result.get("language") or language or "unknown"
        segments = result["segments"]

        # Allineamento Word-Level (se richiesto)
        if enable_word_timestamps and segments:
            try:
                align_model, align_metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=device,
                )
                aligned = whisperx.align(
                    segments, align_model, align_metadata, audio_array, device,
                    return_char_alignments=False,
                )
                segments = aligned["segments"]
                
                # Pulizia approfondita dopo l'allineamento
                del align_model
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Warning: word alignment failed: {e}")

        transcription = format_segments(transcription_mode, segments)
        serialized_segments = serialize_segments(segments)

        # Bug fixato: Creazione corretta della lista dei timestamp
        word_timestamps = [
            {"word": w.get("word", ""), "start": w.get("start", 0), "end": w.get("end", 0)}
            for seg in segments
            for w in seg.get("words", [])
        ] if enable_word_timestamps else None
            
        return detected_language, transcription, serialized_segments, word_timestamps

    def translate(self, format_type, language, batch_size, audio_array):
        trans_result = self._model.transcribe(
            audio_array,
            language=language,
            task="translate",
            batch_size=batch_size,
        )
        return format_segments(format_type, trans_result["segments"])


# --- UTILITIES ---

def serialize_segments(segments):
    return [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        for seg in segments
    ]

def format_segments(format_type, segments):
    if format_type == "plain_text":
        return " ".join(seg["text"].strip() for seg in segments)
    elif format_type == "formatted_text":
        return "\n".join(seg["text"].strip() for seg in segments)
    elif format_type == "srt":
        return write_srt(segments)
    elif format_type == "vtt":
        return write_vtt(segments)
    else:
        return " ".join(seg["text"].strip() for seg in segments)

def _fmt_ts(seconds, always_include_hours=False, decimal_marker="."):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = round((seconds % 1) * 1000)
    if always_include_hours or h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}{decimal_marker}{ms:03d}"
    return f"{m:02d}:{s:02d}{decimal_marker}{ms:03d}"

def write_vtt(segments):
    result = ""
    for seg in segments:
        result += f"{_fmt_ts(seg['start'], always_include_hours=True)} --> {_fmt_ts(seg['end'], always_include_hours=True)}\n"
        result += f"{seg['text'].strip().replace('-->', '->')}\n\n"
    return result

def write_srt(segments):
    result = ""
    for i, seg in enumerate(segments, start=1):
        result += f"{i}\n"
        result += f"{_fmt_ts(seg['start'], always_include_hours=True, decimal_marker=',')} --> "
        result += f"{_fmt_ts(seg['end'], always_include_hours=True, decimal_marker=',')}\n"
        result += f"{seg['text'].strip().replace('-->', '->')}\n\n"
    return result