from dn_audiotranscriber import transcribe_audio, record_audio


data = record_audio()
result = transcribe_audio(data)

print(result["text"])