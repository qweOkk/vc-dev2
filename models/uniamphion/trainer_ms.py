from itertools import chain
import logging
import pickle
import torch
import pyworld as pw
import numpy as np
import soundfile as sf
import os

from torchtts.nn.criterions import GANLoss
from torchtts.nn.criterions import (
    MultiResolutionSTFTLoss,
    MultiResolutionMelSpectrogramLoss,
    SpeakerLoss,
)
from torchtts.nn.metrics import Mean
from torchtts.nn.optim.lr_schedulers import PowerLR, WarmupLR
from torchtts.trainers.base_trainer import Trainer
from torchtts.nn.criterions.duration_loss import DurationPredictorLoss
from torch.optim import Adam
from einops import rearrange

from torchaudio.functional import pitch_shift
from librosa.filters import mel as librosa_mel_fn

logger = logging.getLogger(__name__)

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F




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


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


class NS2Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def interpolate(self, f0):
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            # interpolate the unvoiced f0
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            uv = uv.astype("float")
            uv = np.min(np.array([uv[:-2], uv[1:-1], uv[2:]]), axis=0)
            uv = np.pad(uv, (1, 1))
        return f0, uv

    def extract_world_f0(self, speech):
        audio = speech.cpu().numpy()
        f0s = []
        for i in range(audio.shape[0]):
            wav = audio[i]
            frame_num = len(wav) // 200
            f0, t = pw.dio(wav.astype(np.float64), 16000, frame_period=12.5)
            f0 = pw.stonemask(wav.astype(np.float64), f0, t, 16000)
            f0, _ = self.interpolate(f0)
            f0 = torch.from_numpy(f0).to(speech.device)
            f0s.append(f0[:frame_num])
        f0s = torch.stack(f0s, dim=0).float()
        return f0s

    def train_step(self, batch, acc_step=0):
        batch["pitch"] = self.extract_world_f0(batch["speech"]) # 

        with self.engine.context():
            with torch.no_grad():
                speech = batch["speech"]
                ref_speech = batch["ref_speech"]

                mel = mel_spectrogram(
                    speech,
                    n_fft=1024,
                    num_mels=80,
                    sampling_rate=16000,
                    hop_size=200,
                    win_size=800,
                    fmin=0,
                    fmax=8000,
                )  # (B, 80, T)
                ref_mel = mel_spectrogram(
                    ref_speech,
                    n_fft=1024,
                    num_mels=80,
                    sampling_rate=16000,
                    hop_size=200,
                    win_size=800,
                    fmin=0,
                    fmax=8000,
                )  # (B, 80, T')

                mel = mel.transpose(1, 2)
                ref_mel = ref_mel.transpose(1, 2)

                del speech
                del ref_speech

            if self._config["norm_mel"]:
                mel = -1.0 + 2.0 * (mel - self._config["norm_mel_min"]) / (
                    self._config["norm_mel_max"] - self._config["norm_mel_min"]
                )
                ref_mel = -1.0 + 2.0 * (ref_mel - self._config["norm_mel_min"]) / (
                    self._config["norm_mel_max"] - self._config["norm_mel_min"]
                )

            # print(mel.shape, ref_mel.shape)
            pitch = batch["pitch"]
            # print(pitch.shape)
            duration = batch["duration"]
            # print(duration.shape)
            phone_id = batch["phone_id"]
            # print(phone_id.shape)
            phone_id_mask = batch["phone_id_mask"]
            # print(phone_id_mask.shape)
            mask = batch["mask"]
            # print(mask.shape)
            ref_mask = batch["ref_mask"]
            # print(ref_mask.shape)

            diff_out, prior_out = self.model["generator"](
                x=mel,
                pitch=pitch,
                duration=duration,
                phone_id=phone_id,
                x_ref=ref_mel,
                phone_mask=phone_id_mask,
                x_mask=mask,
                x_ref_mask=ref_mask,
            )

            gen_loss = 0.0

            pitch_loss = log_pitch_loss(prior_out["pitch_pred_log"], pitch, mask=mask)
            gen_loss += pitch_loss
            self.metrics["pitch_loss"].update_state(pitch_loss)

            dur_loss = log_dur_loss(
                prior_out["dur_pred_log"], duration, mask=phone_id_mask
            )
            gen_loss += dur_loss
            self.metrics["dur_loss"].update_state(dur_loss)

            diff_loss_x0 = diff_loss(diff_out["x0_pred"], mel, mask=mask)
            gen_loss += diff_loss_x0
            self.metrics["diff_loss_x0"].update_state(diff_loss_x0)

            # diff loss noise
            diff_loss_noise = diff_loss(
                diff_out["noise_pred"], diff_out["noise"], mask=mask
            )
            gen_loss += diff_loss_noise
            self.metrics["diff_loss_noise"].update_state(diff_loss_noise)

            self.engine.optimize_step(
                loss=gen_loss,
                optimizer=self.optimizers["gen"],
                lr_scheduler=self.lr_schedulers["gen"],
                current_step=acc_step,
                grad_accumulate=self.gradient_accumulate,
                donot_optimize=True if self.gradient_accumulate > 1 else False,
                grad_norm=2e3,
            )

            if self.gradient_accumulate > 1 and acc_step == 0:
                self.engine.optimize_gradients(optimizer=self.optimizers["gen"])

            return {k: m.result() for k, m in self.metrics.items()}

    def configure_optimizers(self):
        gen_params = self.model["generator"].parameters()
        logger.info(
            "generator parameters count: {} M".format(
                sum(
                    p.numel()
                    for p in self.model["generator"].parameters()
                    if p.requires_grad
                )
                / 1e6
            )
        )

        return {
            "gen": Adam(gen_params, **self._config["gen_optim_params"]),
        }

    def configure_lr_schedulers(self):
        """
        return {
            'gen': PowerLR(self.optimizers['gen'],
                           **self._config["gen_schedule_params"]),
        }
        """
        return {
            "gen": WarmupLR(
                self.optimizers["gen"], **self._config["gen_schedule_params"]
            )
        }

    def configure_criteria(self):
        criteria = {
            "l1_loss": torch.nn.L1Loss(reduction="mean"),
            "l2_loss": torch.nn.MSELoss(reduction="mean"),
        }
        return criteria

    def configure_metrics(self):
        metrics = {
            "pitch_loss": Mean(),
            "dur_loss": Mean(),
            "diff_loss_x0": Mean(),
            "diff_loss_noise": Mean(),
        }

        return metrics
