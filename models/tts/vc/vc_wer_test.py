import torch
from transformers import Wav2Vec2Processor, HubertForCTC
import librosa
import numpy as np
import os
import json
import re

#这个是模型
processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")

# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

wav_files_path = "/blob/v-yuancwang/gpt_tts/gpt_tts_melvgan_medium/zero_shot_test/recon198k_librispeech/wav"
# gt_files_path = "/blob/v-yuancwang/gpt_tts/gpt_tts_melvgan/zero_shot_test/recon206k_repeat_3/gt"

# zero_shot_test_json = "/blob/v-shenkai/data/tts/testset/librispeech_test_kai_4_librilight/ref_dur_3_test_dc_1pspk.json"
zero_shot_test_json = "/blob/v-zeqianju/dataset/tts/librispeech/raw/LibriSpeech/ref_dur_3_test_full_with_punc_wdata.json"

test_cases = json.load(open(zero_shot_test_json, "r"))["test_cases"]
uid2text = {}
for test_case in test_cases:
    transcription = test_case["transcription"]
    # upper and remove punctuations
    transcription = transcription.upper()
    transcription = re.sub(r"[^\w\s]", "", transcription)
    uid2text[test_case["uid"]] = transcription

# gt2text = {}
syn2text = {}

for wav_file in os.listdir(wav_files_path):
    utt_id = wav_file.split(".")[0][4 : -len("_generated_e2e")]
    print(utt_id)
    # 怎么转录出text------------------------------
    audio_input, _ = librosa.load(os.path.join(wav_files_path, wav_file), sr=16000)
    audio_input = np.array(audio_input)
    input_values = processor(
        audio_input, return_tensors="pt"
    ).input_values  # Batch size 1
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    print(transcription)
     # 怎么转录出text------------------------------
    syn2text[utt_id] = transcription
    
    # audio_input, _ = librosa.load(os.path.join(gt_files_path, utt_id + ".wav"), sr=16000)
    # audio_input = np.array(audio_input)
    # input_values = processor(audio_input, return_tensors="pt").input_values  # Batch size 1
    # logits = model(input_values).logits
    # predicted_ids = torch.argmax(logits, dim=-1)
    # transcription = processor.decode(predicted_ids[0])
    # print(transcription)
    # gt2text[utt_id] = transcription


# # compute WER between gt2text and uid2text
# from evaluate import load
# wer = load("wer")
# predictions = [v for k, v in sorted(gt2text.items())]
# references = [v for k, v in sorted(uid2text.items())]
# print(predictions)
# print(references)
# wer_score = wer.compute(predictions=predictions, references=references)
# print(wer_score)


# compute WER between syn2text and uid2text
# pi
from evaluate import load
# 用clash 命令开代理
# wer = load("wer")

predictions = [v for k, v in sorted(syn2text.items())]
references = [v for k, v in sorted(uid2text.items())] #ground truth
print(predictions)
print(references)
wer_score = wer.compute(predictions=predictions, references=references)
print(wer_score)
