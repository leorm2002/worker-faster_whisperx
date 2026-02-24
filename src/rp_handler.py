import base64
import tempfile
import os

from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict

# Inizializzazione globale del modello
MODEL = predict.Predictor()
MODEL.setup()

def base64_to_tempfile(base64_file: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))
    return temp_file.name


@rp_debugger.FunctionTimer
def run_whisper_job(job):
    job_input = job['input']

    # 1. Validazione Input
    with rp_debugger.LineTimer('validation_step'):
        input_validation = validate(job_input, INPUT_VALIDATIONS)
        if 'errors' in input_validation:
            return {"error": input_validation['errors']}
        job_input = input_validation['validated_input']

    # 2. Controllo Logico Audio
    if not job_input.get('audio') and not job_input.get('audio_base64'):
        return {'error': 'Must provide either audio or audio_base64'}

    if job_input.get('audio') and job_input.get('audio_base64'):
        return {'error': 'Must provide either audio or audio_base64, not both'}

    # 3. Preparazione File Audio
    audio_input = None
    is_temp_file = False

    if job_input.get('audio'):
        with rp_debugger.LineTimer('download_step'):
            audio_input = download_files_from_urls(job['id'], [job_input['audio']])[0]

    if job_input.get('audio_base64'):
        audio_input = base64_to_tempfile(job_input['audio_base64'])
        is_temp_file = True

    # 4. Esecuzione Predizione
    with rp_debugger.LineTimer('prediction_step'):
        try:
            whisper_results = MODEL.predict(
                audio=audio_input,
                model_name=job_input.get("model", predict.DEFAULT_MODEL),
                transcription_mode=job_input.get("transcription", predict.DEFAULT_TRANSCRIPTION_FORMAT),
                do_translate=job_input.get("translate", False),
                translation_format=job_input.get("translation", predict.DEFAULT_TRANSLATION_FORMAT),
                language=job_input.get("language"),
                beam_size=job_input.get("beam_size", 5),
                initial_prompt=job_input.get("initial_prompt"),
                condition_on_previous_text=job_input.get("condition_on_previous_text", False),
                logprob_threshold=job_input.get("logprob_threshold", -1.0),
                no_speech_threshold=job_input.get("no_speech_threshold", 0.6),
                suppress_numerals=job_input.get("suppress_numerals", False),
                hotwords=job_input.get("hotwords"),
                enable_word_timestamps=job_input.get("word_timestamps", False),
                batch_size=job_input.get("batch_size", predict.DEFAULT_BATCH_SIZE),
                vad_method=job_input.get("vad_method", predict.DEFAULT_VAD_METHOD),
            )
        finally:
            # Assicuriamoci che il file temporaneo venga eliminato anche in caso di errore
            if is_temp_file and os.path.exists(audio_input):
                os.remove(audio_input)

    # 5. Pulizia standard di RunPod
    with rp_debugger.LineTimer('cleanup_step'):
        rp_cleanup.clean(['input_objects'])

    return whisper_results


runpod.serverless.start({"handler": run_whisper_job})