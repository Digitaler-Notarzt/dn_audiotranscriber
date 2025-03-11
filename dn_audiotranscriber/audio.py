import numpy as np
import librosa

def prepare_pcm16_audio_whisper(pcm16_data, sample_rate, target_sample_rate=16000):
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

def normalize_audio_whisper(audio_array):
    """
    Normalize an audio array for Whisper:
    - Center the audio by subtracting the mean.
    - Scale so the maximum absolute value is 1.
    - Convert to float32.
    
    Parameters:
        audio_array (array-like): Input audio data as a 1-D array.
    
    Returns:
        np.ndarray: Normalized audio array as float32.
    """
    # Ensure the input is a numpy array
    audio_array = np.asarray(audio_array)
    
    # Step 1: Center the audio by subtracting the mean
    mean_val = np.mean(audio_array)
    centered_array = audio_array - mean_val
    
    # Step 2: Find the maximum absolute value and scale
    max_val = np.max(np.abs(centered_array))
    if max_val > 0:  # Avoid division by zero
        normalized_array = centered_array / max_val
    else:
        normalized_array = centered_array  # If all zeros, return as is
    
    # Step 3: Convert to float32
    normalized_array = normalized_array.astype(np.float32)
    
    return normalized_array
