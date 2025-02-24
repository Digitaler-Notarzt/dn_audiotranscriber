import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import librosa
from datasets import load_dataset

import logging

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self, model_id = "primeline/whisper-large-v3-german"):
        logger.info("Initializing Whisper transcriber with model_id: ", model_id)
        if torch.cuda.is_available():
            self.device = "cuda:0"
        elif torch.mps.is_available():
            self.device = "mps:0"
        else:
            self.device = "cpu"

        logger.info("Whisper ", model_id)
        
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(self.device)
        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=self.device,
        )

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

    def transcribe(self, audio: np.ndarray):
        self.pipe(audio)
