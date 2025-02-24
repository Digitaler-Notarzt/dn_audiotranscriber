import pytest
from dn_audiotranscriber import WhisperTranscriber
from datasets import load_dataset

def test_whisperlocal():
    wt = WhisperTranscriber()
    dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")

    sample = dataset[0]["audio"]
    result = wt.pipe(sample)

    print(result["text"])

