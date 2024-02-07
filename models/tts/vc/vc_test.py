# Copyright 2023
# Author     : Dongchao Yang
# Description: Compute speaker similarity score

import os
import glob
import argparse
from tqdm import tqdm
from scipy.io import wavfile

# from pystoi import stoi
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
import torch
import scipy.signal as signal

# from datasets import load_dataset
import soundfile as sf
import librosa

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/mnt/data2/hehaorui/ckpt/wavlm")
model = WavLMForXVector.from_pretrained("/mnt/data2/hehaorui/ckpt/wavlm")

# feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
# model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
device = torch.device("cuda:5")
model = model.to(device)


def WavLM_SV(ge_audio, ref_audio):
    audio = [ge_audio, ref_audio]
    inputs = feature_extractor(audio, padding=True, sampling_rate=16000, return_tensors="pt")
    for key in inputs.keys():
        inputs[key] = inputs[key].to(device)
    with torch.no_grad():
        embeddings = model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        similarity = cosine_sim(embeddings[0], embeddings[1])
    return similarity


def calculate_speaker_similarity(ref_dir, deg_dir):
    deg_files = glob.glob(f"{deg_dir}/*.wav")
    if len(deg_files) < 1:
        raise RuntimeError(f"Found no wavs in {deg_dir}")
    similarity_scores = []
    for deg_wav in tqdm(deg_files):
        ref_wav = os.path.join(ref_dir, os.path.basename(deg_wav))  #
        if os.path.exists(ref_wav) == False:
            continue
        ref, _ = librosa.load(ref_wav, sr=16000)
        deg, _ = librosa.load(deg_wav, sr=16000)
        similarity = WavLM_SV(deg, ref)
        similarity_scores.append(similarity.item())
    return np.mean(similarity_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute speaker similarity_score")
    parser.add_argument(
        "-r", "--ref_dir", required=True, help="Reference wave folder or file list."
    )
    parser.add_argument("-d", "--deg_dir", required=True, help="Degraded wave folder.")
    args = parser.parse_args()
    similarity_score = calculate_speaker_similarity(args.ref_dir, args.deg_dir)
    print(f"Speaker similarity: {similarity_score}")
