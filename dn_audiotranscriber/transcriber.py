import numpy as np
import whisper

def transcribe_audio(filepath: str):
    model = whisper.load_model("base")

    audio = whisper.load_audio(filepath)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio=audio, n_mels=128).to(model.device)
    _, probs = model.detect_language(mel)

    print(f"Detected language: {max(probs, key=probs.get)}")

    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)
    return result