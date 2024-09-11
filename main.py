from dn_audiotranscriber import transcribe_audio 
from soundrecorder import record_audio, save_audio, delete_audio


audio_data = record_audio()
save_audio(audio_data, "test.wav")
try:
    result = transcribe_audio("test.wav")
    print(result.text)
finally:
    delete_audio("test.wav")