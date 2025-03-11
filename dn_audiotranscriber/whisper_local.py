import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import librosa
from datasets import load_dataset
import warnings
import logging

warnings.filterwarnings("ignore", message="`max_new_tokens` is deprecated")
warnings.filterwarnings("ignore", message="`inputs` is deprecated")

logger = logging.getLogger(__name__)

class WhisperTranscriber:
    def __init__(self, model_id = "primeline/whisper-large-v3-turbo-german", cpu = False):
        logger.info("Initializing Whisper transcriber with model_id: %s", model_id)

        if torch.cuda.is_available():
            self.device = "cuda:0"
        elif torch.mps.is_available():
            self.device = "mps:0"
        else:
            self.device = "cpu"

        if cpu: self.device =  "cpu"

        logger.info("Using device: %s", self.device)
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

    def transcribe(self, audio: np.ndarray):
        """
        Transcribe audio with noise suppression.
        It is very important that the audio is already sampled in 16khz
        and normalized to -1 to 1.

        Args:
            audio (np.ndarray): Input audio array.
            sample_rate (int): Sample rate of the audio.

        Returns:
            dict: Transcription result
        """

        result = self.pipe(audio)
        return result
