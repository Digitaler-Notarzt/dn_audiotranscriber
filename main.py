#!/usr/bin/env python3

from dn_audiotranscriber import transcribe_audio, prepare_pcm16_audio
from scipy.io.wavfile import write
import sounddevice as sd
import os
import logging
from dn_base import dinologger
import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")


logger = dinologger.get_logger("dinologger", level=logging.DEBUG, log_file="app.log")
seconds = 5
dir_path = os.patseconds = 5
freq = 44100

def record_audio():
    audio_data = sd.rec(int(seconds * freq), samplerate=freq, channels=1, dtype='int16')
    sd.wait()
    return audio_data

print("Recording now")
audio = record_audio()
print("Transcribing")
audio = prepare_pcm16_audio(audio, freq)
result = transcribe_audio(audio)

print(f'Resulting Text: {result["text"]}')
