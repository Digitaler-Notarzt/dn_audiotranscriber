import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

import numpy as np
import librosa

import logging
from dn_base import dinologger

logger = dinologger.get_logger("dinologger")

if torch.cuda.is_available():
    device = "cuda:0"
elif torch.mps.is_available():
    device = "mps:0"
else:
    device = "cpu"

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

logger.log(msg=f'Using device: {device}', level=logging.DEBUG)

model_id = "primeline/whisper-large-v3-turbo-german"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


def transcribe_audio(audio: np.ndarray):
    return pipe(audio)

def prepare_pcm16_audio(pcm16_data, sample_rate, target_sample_rate=16000):
    """
    Prepares PCM16 audio data for Whisper.

    Args:
        pcm16_data (np.ndarray): PCM16 audio data as a 1D or 2D NumPy array.
        sample_rate (int): Original sample rate of the audio.
        target_sample_rate (int): Target sample rate (default 16 kHz for Whisper).

    Returns:
        np.ndarray: Audio data ready for Whisper (1D float32 array at 16 kHz).
    """
    # Ensure data is in a NumPy array
    pcm16_data = np.asarray(pcm16_data)

    # Flatten multi-channel audio to mono (e.g., stereo to single channel)
    if pcm16_data.ndim > 1:
        pcm16_data = np.mean(pcm16_data, axis=1)

    # Convert PCM16 (int16) to float32 and normalize to -1.0 to 1.0
    audio_float32 = pcm16_data.astype(np.float32) / 32768.0

    if sample_rate != target_sample_rate:
        audio_resampled = librosa.resample(audio_float32, orig_sr=sample_rate, target_sr=target_sample_rate)
    else:
        audio_resampled = audio_float32

    return audio_resampled