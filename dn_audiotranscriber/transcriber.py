import whisper
import numpy as np
import librosa

def transcribe_audio(audio: np.ndarray):
    model = whisper.load_model("tiny")
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio=audio).to(model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    return result

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