import os

import sounddevice as sd
from scipy.io import wavfile
import numpy as np

seconds = 5
sample_rate = 44100  # Sample rate for recording
channels = 1         # Number of audio channels (1 for mono, 2 for stereo)

def record_audio():
    
    # A list to store the audio chunks
    recorded_data = sd.rec(frames=int(seconds*sample_rate), samplerate=sample_rate, channels=channels)
    sd.wait()

    # Convert the list of chunks into a numpy array
    audio_data = np.concatenate(recorded_data, axis=0)
    
    return audio_data

def save_audio(audio_data, filename):
    # Scale the data to be in the range of int16
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Write the audio data to a file
    wavfile.write(filename, sample_rate, audio_data)
    print(f"Audio saved to {filename}")

def delete_audio(filename):
    os.remove(filename)
    print(f"Audio file {filename} deleted")