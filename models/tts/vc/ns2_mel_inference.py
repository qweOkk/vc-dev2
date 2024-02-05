import os
import time
import sys
import json
import transformers

from models.tts.vc.ns2_uniamphion import UniAmphionVC
from models.tts.vc.vc_dataset import  VCCollator, VCDataset, batch_by_size
from models.tts.vc.hubert_kmeans import HubertWithKmeans

APP_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir
    )
)
sys.path.insert(0, APP_ROOT)


from hubert_kmeans import HubertWithKmeans

import hydra
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

from torchtts import models
import codecs
import soundfile as sf
import matplotlib.pyplot as plt
import zipfile

import numpy as np
import tarfile
import pickle
from collections import defaultdict

import soundfile as sf
import librosa
import librosa.display
import io
import copy

from librosa.util import normalize
from librosa.filters import mel as librosa_mel_fn
from torchaudio.functional import pitch_shift

import torchaudio

import pyworld as pw

logger = logging.getLogger(__name__)

sr = 16000

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}
init_mel_and_hann = False


def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window, init_mel_and_hann
    if not init_mel_and_hann:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
        print(mel_basis)
        init_mel_and_hann = True

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def interpolate(f0):
    uv = f0 == 0
    if len(f0[~uv]) > 0:
        # interpolate the unvoiced f0
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        uv = uv.astype("float")
        uv = np.min(np.array([uv[:-2], uv[1:-1], uv[2:]]), axis=0)
        uv = np.pad(uv, (1, 1))
    return f0, uv


def extract_world_f0(speech):
    audio = speech.cpu().numpy()
    f0s = []
    for i in range(audio.shape[0]):
        wav = audio[i]
        frame_num = len(wav) // 200
        f0, t = pw.dio(wav.astype(np.float64), 16000, frame_period=12.5)
        f0 = pw.stonemask(wav.astype(np.float64), f0, t, 16000)
        f0, _ = interpolate(f0)
        f0 = torch.from_numpy(f0).to(speech.device)
        f0s.append(f0[:frame_num])
    f0s = torch.stack(f0s, dim=0).float()
    return f0s


class Wav2vec2(torch.nn.Module):
    def __init__(self, layer=12):
        super().__init__()
        self.wav2vec2 = transformers.Wav2Vec2ForPreTraining.from_pretrained(
            "facebook/wav2vec2-xls-r-300m"
        )
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
            param.grad = None
        self.wav2vec2.eval()
        self.feature_layer = layer

    @torch.no_grad()
    def forward(self, x):
        x = F.pad(x, (40, 40), "reflect")
        outputs = self.wav2vec2(x.squeeze(1), output_hidden_states=True)
        y = outputs.hidden_states[self.feature_layer]
        y = y.permute((0, 2, 1))
        y = F.interpolate(y, scale_factor=8, mode="nearest")
        y = F.interpolate(y, scale_factor=0.2, mode="nearest")
        y = y.permute((0, 2, 1))
        return y


@hydra.main(config_path="../../configs", config_name="train")
def main(config):
    with torch.inference_mode():
        print(config)
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        model = models.build_model(config.model)

        checkpoint = torch.load(config.checkpoint_path, map_location="cpu")

        for k, v in model.items():
            v.load_state_dict(checkpoint["model"][k])
            v = v.to(device).eval()

            pytorch_total_params = sum(p.numel() for p in v.parameters())
            logger.info(
                "{} pytorch_total_params: {} M".format(k, pytorch_total_params / 1e6)
            )

        # w2v = Wav2vec2()
        # w2v = w2v.to(device)
        w2v = HubertWithKmeans(
            checkpoint_path="/blob/v-yuancwang/amphion_ns2/mhubert_base_vp_en_es_fr_it3.pt",
            kmeans_path="/blob/v-yuancwang/amphion_ns2/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin",
        )
        w2v = w2v.to(device)

        zero_shot_json_file_path = (
            "/blob/v-yuancwang/codec_ckpt/vc_test/test.json"
        )
        with open(zero_shot_json_file_path, "r") as f:
            zero_shot_json = json.load(f)
        zero_shot_json = zero_shot_json["test_cases"]
        print(len(zero_shot_json))

        utt_dict = {}
        for info in zero_shot_json:
            utt_id = info["uid"]
            utt_dict[utt_id] = {}
            utt_dict[utt_id]["source_speech"], _ = librosa.load(
                info["source_wav_path"], sr=16000
            )
            utt_dict[utt_id]["target_speech"], _ = librosa.load(
                info["target_wav_path"], sr=16000
            )
            utt_dict[utt_id]["prompt_speech"], _ = librosa.load(
                info["prompt_wav_path"], sr=16000
            )

        os.makedirs(config.output, exist_ok=True)
        test_cases = []

        temp_id = 0
        for utt_id, utt in utt_dict.items():
            # if temp_id > 40:
            #     break
            temp_id += 1
            print(utt_id)

            wav = utt["source_speech"]
            wav = np.pad(wav, (0, 1600 - len(wav) % 1600))
            audio = torch.from_numpy(wav).to(device)
            audio = audio[None, :]
            print(audio.shape)

            tgt_wav = utt["target_speech"]
            tgt_wav = np.pad(tgt_wav, (0, 1600 - len(tgt_wav) % 1600))
            tgt_audio = torch.from_numpy(tgt_wav).to(device)
            tgt_audio = tgt_audio[None, :]
            print(tgt_audio.shape)

            ref_wav = utt["prompt_speech"]
            ref_wav = np.pad(ref_wav, (0, 200 - len(ref_wav) % 200))
            ref_audio = torch.from_numpy(ref_wav).to(device)
            ref_audio = ref_audio[None, :]
            print(ref_audio.shape)

            with torch.no_grad():
                ref_mel = mel_spectrogram(
                    ref_audio,
                    n_fft=1024,
                    num_mels=80,
                    sampling_rate=16000,
                    hop_size=200,
                    win_size=800,
                    fmin=0,
                    fmax=8000,
                )
                tgt_mel = mel_spectrogram(
                    tgt_audio,
                    n_fft=1024,
                    num_mels=80,
                    sampling_rate=16000,
                    hop_size=200,
                    win_size=800,
                    fmin=0,
                    fmax=8000,
                )
                source_mel = mel_spectrogram(
                    audio,
                    n_fft=1024,
                    num_mels=80,
                    sampling_rate=16000,
                    hop_size=200,
                    win_size=800,
                    fmin=0,
                    fmax=8000,
                )
                ref_mel = ref_mel.transpose(1, 2).to(device=device)
                print(ref_mel.shape)
                ref_mask = torch.ones(ref_mel.shape[0], ref_mel.shape[1]).to(device)

      
                _, content_feature = w2v(audio)
                content_feature = content_feature.to(device=device)

                pitch = extract_world_f0(audio).to(device=device)
                pitch = (pitch - pitch.mean(dim=1, keepdim=True)) / (
                    pitch.std(dim=1, keepdim=True) + 1e-6
                )


            x0 = model["generator"].inference(
                content_feature=content_feature,
                pitch=pitch,
                # x_ref=torch.zeros(ref_mel.shape).to(ref_mel.device),
                x_ref=ref_mel,
                x_ref_mask=ref_mask,
                inference_steps=200,
                sigma=1.2,
            )

            test_case = dict()
            os.makedirs(f"{config.output}/recon/mel", exist_ok=True)
            os.makedirs(f"{config.output}/target/mel", exist_ok=True)
            os.makedirs(f"{config.output}/source/mel", exist_ok=True)
            os.makedirs(f"{config.output}/prompt/mel", exist_ok=True)
            recon_path = f"{config.output}/recon/mel/{utt_id}.npy"
            ref_path = f"{config.output}/target/mel/{utt_id}.npy"
            source_path = f"{config.output}/source/mel/{utt_id}.npy"
            prompt_path = f"{config.output}/prompt/mel/{utt_id}.npy"
            test_case["recon_ref_wav_path"] = recon_path.replace(
                "/mel/", "/wav/"
            ).replace(".npy", "_generated_e2e.wav")
            test_case["reference_wav_path"] = ref_path.replace(
                "/mel/", "/wav/"
            ).replace(".npy", "_generated_e2e.wav")
            np.save(recon_path, x0.transpose(1, 2).detach().cpu().numpy())
            np.save(prompt_path, ref_mel.transpose(1, 2).detach().cpu().numpy())
            np.save(ref_path, tgt_mel.detach().cpu().numpy())
            np.save(source_path, source_mel.detach().cpu().numpy())
            test_cases.append(test_case)
        data = dict()
        data["dataset"] = "recon"
        data["test_cases"] = test_cases
        with open(f"{config.output}/recon.json", "w") as f:
            json.dump(data, f)

        os.system(
            f"python /home/hehaorui/code/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output}/recon/mel'} --output_dir={f'{args.output}/recon/wav'} --checkpoint_file=/mnt/data2/wangyuancheng/ns2_ckpts/bigvgan/g_00490000"
        )
        os.system(
            f"python /home/hehaorui/code/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output}/target/mel'} --output_dir={f'{args.output}/target/wav'} --checkpoint_file=/mnt/data2/wangyuancheng/ns2_ckpts/bigvgan/g_00490000"
        )
        os.system(
            f"python /home/hehaorui/code/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output}/source/mel'} --output_dir={f'{args.output}/source/wav'} --checkpoint_file=/mnt/data2/wangyuancheng/ns2_ckpts/bigvgan/g_00490000"
        )
        os.system(
            f"python /home/hehaorui/code/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output}/prompt/mel'} --output_dir={f'{args.output}/prompt/wav'} --checkpoint_file=/mnt/data2/wangyuancheng/ns2_ckpts/bigvgan/g_00490000"
        )

        os.system(f"python /home/t-zeqianju/torchtts_noamlt/test_vc.py -r={f'{config.output}/prompt/wav'} -d={f'{config.output}/recon/wav'}")
        os.system(f"python /home/t-zeqianju/torchtts_noamlt/test_vc.py -r={f'{config.output}/target/wav'} -d={f'{config.output}/recon/wav'}")
        os.system(f"python /home/t-zeqianju/torchtts_noamlt/test_vc.py -r={f'{config.output}/prompt/wav'} -d={f'{config.output}/target/wav'}")        

if __name__ == "__main__":
    main()
