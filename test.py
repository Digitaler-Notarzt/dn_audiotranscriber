#!/usr/bin/env python3

from dn_audiotranscriber import WhisperTranscriber
from scipy.io.wavfile import write
import os
import sys
import logging
import warnings
import torch
from datasets import load_dataset
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

import logging

logging.basicConfig(level=logging.INFO)

num_samples_to_take = 1

wt = WhisperTranscriber()
dataset = load_dataset("avemio/ASR-GERMAN-MIXED-TEST", split="test", streaming=False)

for datasample in dataset:
    audio_data = datasample["audio"]["array"]
    normalized_audio = WhisperTranscriber.normalize_audio(audio_data)

    result = wt.transcribe(normalized_audio)

    print(result)

sys.exit(0)

#for example in dataset:
    #print(example["audio"])
    # sample = example["audio"]
    # result = wt.pipe(sample)

    # print(f'Resulting Text: {result["text"]}')
    # print(f'Targetc Text: {dataset[0]["transkription"]}')
#    pass

