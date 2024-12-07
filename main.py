from dn_audiotranscriber import transcribe_audio, prepare_pcm16_audio
from scipy.io.wavfile import write
import sounddevice as sd
import os
import numpy as np
import librosa
import logging
from dn_base import dinologger

logger = dinologger.get_logger("dinologger", level=logging.DEBUG, log_file="app.log")

import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

seconds = 5
dir_path = os.patseconds = 5
freq = 44100

def record_audio():
    audio_data = sd.rec(int(seconds * freq), samplerate=freq, channels=1, dtype='int16')
    logger.log(msg="Starting recording", level=logging.DEBUG)
    sd.wait()
    logger.log(msg="Finished recording", level=logging.DEBUG)
    return audio_data


audio = record_audio()

logger.log(msg="Preparing audio", level=logging.DEBUG)
audio = prepare_pcm16_audio(audio, freq)

logger.log(msg=f"Audio shape after preparation: {audio.shape}", level=logging.DEBUG)

logger.log(msg="Transcribing audio....", level=logging.DEBUG)
result = transcribe_audio(audio)

logger.log(msg=result["text"], level=logging.DEBUG)
