import numpy as np
import whisper

from .soundrecorder import save_audio, delete_audio

filename = "temp.wav"

def transcribe_audio(audio_data):

    # Save the recorded audio to a WAV file
    save_audio(audio_data, filename)
    try:
        print("Finished recording")

        model = whisper.load_model("base")

        audio = whisper.load_audio(filename)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio=audio, n_mels=128).to(model.device)
        _, probs = model.detect_language(mel)

        print(f"Detected language: {max(probs, key=probs.get)}")

        options = whisper.DecodingOptions()
        result = whisper.decode(model, mel, options)
        return result
    finally:
        delete_audio(filename)