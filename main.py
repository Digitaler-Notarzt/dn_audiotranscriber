from dn_audiotranscriber import transcribe_audio, prepare_pcm16_audio
from scipy.io.wavfile import write
import sounddevice as sd
import os
import numpy as np
import librosa

import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

seconds = 5
dir_path = os.patseconds = 5
freq = 44100

audio_data = sd.rec(int(seconds * freq), samplerate=freq, channels=1, dtype='int16')
sd.wait()

print("Finished recording")

audio_data = prepare_pcm16_audio(audio_data, freq)
result = transcribe_audio(audio_data)
print(result.text)