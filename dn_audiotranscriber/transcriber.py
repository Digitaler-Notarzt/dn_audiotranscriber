import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import whisper

seconds = 5
sample_rate = 44100  # Sample rate for recording
channels = 1         # Number of audio channels (1 for mono, 2 for stereo)

filename = "input_audio.wav"

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

def transcribe_audio():
    # Record the audio
    audio_data = record_audio()
    # Save the recorded audio to a WAV file
    save_audio(audio_data, filename)
    print("Finished recording")

    model = whisper.load_model("large-v3")

    audio = whisper.load_audio(filename)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio=audio, n_mels=128).to(model.device)
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    return result.text