# worker-faster_whisperx

Worker RunPod per trascrizione audio con WhisperX.

## Modelli supportati

- `tiny`
- `turbo`
- `large-v2`

Default: `large-v2`

## Input

| Campo | Tipo | Note |
| --- | --- | --- |
| `audio` | `str` | URL del file audio |
| `audio_base64` | `str` | Audio codificato in base64 |
| `model` | `str` | `tiny`, `turbo`, `large-v2` |
| `transcription` | `str` | `plain_text`, `formatted_text`, `srt`, `vtt` |
| `translate` | `bool` | Se `true`, traduce in inglese |
| `translation` | `str` | `plain_text`, `formatted_text`, `srt`, `vtt` |
| `language` | `str` | Lingua esplicita oppure `auto`/`null`/omesso per auto-detect |
| `beam_size` | `int` | Default `5` |
| `initial_prompt` | `str` | Prompt iniziale opzionale |
| `condition_on_previous_text` | `bool` | Default `false` |
| `logprob_threshold` | `float` | Default `-1.0` |
| `no_speech_threshold` | `float` | Default `0.6` |
| `suppress_numerals` | `bool` | Default `false` |
| `hotwords` | `str` | Hotwords opzionali |
| `word_timestamps` | `bool` | Include i timestamp parola per parola |
| `batch_size` | `int` | Default `16` |
| `vad_method` | `str` | `silero` o `pyannote` |

Note:
- fornire uno solo tra `audio` e `audio_base64`
- `language: "auto"` attiva il rilevamento automatico

## Output

Il worker restituisce:

- `segments`
- `detected_language`
- `transcription`
- `translation`
- `word_timestamps`
- `device`
- `model`

## Esempio

```json
{
  "input": {
    "audio": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav",
    "model": "turbo",
    "language": "auto",
    "transcription": "plain_text"
  }
}
```
