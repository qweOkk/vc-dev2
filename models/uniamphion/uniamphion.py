import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import json5
import os
from librosa.filters import mel as librosa_mel_fn
from DiffTransformer import DiffTransformer
from PriorEncoder import PriorEncoder
from WaveNet import DiffWaveNet
from ReferenceEncoder import ReferenceEncoder

sr = 16000 # sampling rate
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


class Diffusion(nn.Module):
    def __init__(self, cfg, diff_model):
        super().__init__()

        self.cfg = cfg
        self.diff_estimator = diff_model
        self.beta_min = cfg.beta_min
        self.beta_max = cfg.beta_max
        self.sigma = cfg.sigma
        self.noise_factor = cfg.noise_factor

    def forward(self, x, condition_embedding, x_mask, reference_embedding, offset=1e-5):
        diffusion_step = torch.rand(
            x.shape[0], dtype=x.dtype, device=x.device, requires_grad=False
        )
        diffusion_step = torch.clamp(diffusion_step, offset, 1.0 - offset)
        xt, z = self.forward_diffusion(x0=x, diffusion_step=diffusion_step)

        cum_beta = self.get_cum_beta(diffusion_step.unsqueeze(-1).unsqueeze(-1))
        x0_pred = self.diff_estimator(
            xt, condition_embedding, x_mask, reference_embedding, diffusion_step
        )
        mean_pred = x0_pred * torch.exp(-0.5 * cum_beta / (self.sigma**2))
        variance = (self.sigma**2) * (1.0 - torch.exp(-cum_beta / (self.sigma**2)))
        noise_pred = (xt - mean_pred) / (torch.sqrt(variance) * self.noise_factor)
        noise = z
        diff_out = {"x0_pred": x0_pred, "noise_pred": noise_pred, "noise": noise}
        return diff_out

    @torch.no_grad()
    def get_cum_beta(self, time_step):
        return self.beta_min * time_step + 0.5 * (self.beta_max - self.beta_min) * (
            time_step**2
        )

    @torch.no_grad()
    def get_beta_t(self, time_step):
        return self.beta_min + (self.beta_max - self.beta_min) * time_step

    @torch.no_grad()
    def forward_diffusion(self, x0, diffusion_step):
        time_step = diffusion_step.unsqueeze(-1).unsqueeze(-1)
        cum_beta = self.get_cum_beta(time_step)
        mean = x0 * torch.exp(-0.5 * cum_beta / (self.sigma**2))
        variance = (self.sigma**2) * (1 - torch.exp(-cum_beta / (self.sigma**2)))
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt = mean + z * torch.sqrt(variance) * self.noise_factor
        return xt, z

    @torch.no_grad()
    def cal_dxt(
        self, xt, condition_embedding, x_mask, reference_embedding, diffusion_step, h
    ):
        time_step = diffusion_step.unsqueeze(-1).unsqueeze(-1)
        cum_beta = self.get_cum_beta(time_step=time_step)
        beta_t = self.get_beta_t(time_step=time_step)
        x0_pred = self.diff_estimator(
            xt, condition_embedding, x_mask, reference_embedding, diffusion_step
        )
        mean_pred = x0_pred * torch.exp(-0.5 * cum_beta / (self.sigma**2))
        noise_pred = xt - mean_pred
        variance = (self.sigma**2) * (1.0 - torch.exp(-cum_beta / (self.sigma**2)))
        logp = -noise_pred / (variance + 1e-8)
        dxt = -0.5 * h * beta_t * (logp + xt / (self.sigma**2))
        return dxt

    @torch.no_grad()
    def reverse_diffusion(
        self, z, condition_embedding, x_mask, reference_embedding, n_timesteps
    ):
        h = 1.0 / max(n_timesteps, 1)
        xt = z
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )
            dxt = self.cal_dxt(
                xt,
                condition_embedding,
                x_mask,
                reference_embedding,
                diffusion_step=t,
                h=h,
            )
            xt_ = xt - dxt
            if self.cfg.ode_solve_method == "midpoint":
                x_mid = 0.5 * (xt_ + xt)
                dxt = self.cal_dxt(
                    x_mid,
                    condition_embedding,
                    x_mask,
                    reference_embedding,
                    diffusion_step=t + 0.5 * h,
                    h=h,
                )
                xt = xt - dxt
            elif self.cfg.ode_solve_method == "euler":
                xt = xt_
        return xt

    @torch.no_grad()
    def reverse_diffusion_from_t(
        self, z, condition_embedding, x_mask, reference_embedding, n_timesteps, t_start
    ):
        h = t_start / max(n_timesteps, 1)
        xt = z
        for i in range(n_timesteps):
            t = (t_start - (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )
            dxt = self.cal_dxt(
                xt,
                x_mask,
                condition_embedding,
                x_mask,
                reference_embedding,
                diffusion_step=t,
                h=h,
            )
            xt_ = xt - dxt
            if self.cfg.ode_solve_method == "midpoint":
                x_mid = 0.5 * (xt_ + xt)
                dxt = self.cal_dxt(
                    x_mid,
                    condition_embedding,
                    x_mask,
                    reference_embedding,
                    diffusion_step=t + 0.5 * h,
                    h=h,
                )
                xt = xt - dxt
            elif self.cfg.ode_solve_method == "euler":
                xt = xt_
        return xt


class UniAmphionBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.reference_encoder = ReferenceEncoder(cfg=cfg.reference_encoder)
        self.diffusion = Diffusion(
            cfg=cfg.diffusion,
            diff_model=DiffTransformer(cfg=cfg.diffusion.diff_transformer),
        )


    def forward(
        self, x, condition_embedding=None, x_mask=None, x_ref=None, x_ref_mask=None
    ):
        reference_embedding, _ = self.reference_encoder(
            x_ref=x_ref, key_padding_mask=x_ref_mask
        )

        diff_out = self.diffusion(x, condition_embedding, x_mask, reference_embedding)

        return diff_out

    @torch.no_grad()
    def inference(
        self,
        condition_embedding=None,
        x_ref=None,
        x_ref_mask=None,
        inference_steps=1000,
        sigma=1.2,
    ):
        bsz, l, _ = condition_embedding.shape
        z = (
            torch.randn(bsz, l, self.cfg.diffusion.diff_transformer.in_dim).to(
                condition_embedding.device
            )
            / sigma
        )

        reference_embedding, _ = self.reference_encoder(
            x_ref=x_ref, key_padding_mask=x_ref_mask
        )

        x0 = self.diffusion.reverse_diffusion(
            z=z,
            condition_embedding=condition_embedding,
            x_mask=None,
            reference_embedding=reference_embedding,
            n_timesteps=inference_steps,
        )

        return x0

    @torch.no_grad()
    def reverse_diffusion_from_t(
        self,
        x,
        condition_embedding=None,
        x_mask=None,
        x_ref=None,
        x_ref_mask=None,
        inference_steps=None,
        t=None,
    ):
        reference_embedding, _ = self.reference_encoder(
            x_ref=x_ref, key_padding_mask=x_ref_mask
        )

        diffusion_step = (
            torch.ones(
                x.shape[0],
                dtype=x.dtype,
                device=x.device,
                requires_grad=False,
            )
            * t
        )
        diffusion_step = torch.clamp(diffusion_step, 1e-5, 1.0 - 1e-5)
        xt, _ = self.diffusion.forward_diffusion(x0=x, diffusion_step=diffusion_step)

        x0 = self.diffusion.reverse_diffusion_from_t(
            z=xt,
            condition_embedding=condition_embedding,
            x_mask=x_mask,
            reference_embedding=reference_embedding,
            n_timesteps=inference_steps,
            t_start=t,
        )

        return x0


class UniAmphionTTS(nn.Module):
    def __init__(self, cfg_path):
        super().__init__()

        cfg = load_config(cfg_path)
        cfg = cfg.model
        self.cfg = cfg

        self.reference_encoder = ReferenceEncoder(cfg=cfg.reference_encoder)
        if cfg.diffusion.diff_model_type == "Transformer":
            self.diffusion = Diffusion(
                cfg=cfg.diffusion,
                diff_model=DiffTransformer(cfg=cfg.diffusion.diff_transformer),
            )
        elif cfg.diffusion.diff_model_type == "WaveNet":
            self.diffusion = Diffusion(
                cfg=cfg.diffusion,
                diff_model=DiffWaveNet(cfg=cfg.diffusion.diff_wavenet),
            )
        # TODO: add error raise

        self.prior_encoder = PriorEncoder(cfg=cfg.prior_encoder)

    def forward(
        self,
        x=None,
        pitch=None,
        duration=None,
        phone_id=None,
        x_ref=None,
        phone_mask=None,
        x_mask=None,
        x_ref_mask=None,
    ):
        reference_embedding, reference_latent = self.reference_encoder(
            x_ref=x_ref, key_padding_mask=x_ref_mask
        )

        prior_out = self.prior_encoder(
            phone_id=phone_id,
            duration=duration,
            pitch=pitch,
            phone_mask=phone_mask,
            mask=x_mask,
            ref_emb=reference_latent,
            ref_mask=x_ref_mask,
            is_inference=False,
        )

        condition_embedding = prior_out["prior_out"]

        diff_out = self.diffusion(
            x=x,
            condition_embedding=condition_embedding,
            x_mask=x_mask,
            reference_embedding=reference_embedding,
        )

        return diff_out, prior_out

    @torch.no_grad()
    def inference(
        self,
        phone_id=None,
        x_ref=None,
        x_ref_mask=None,
        inference_steps=1000,
        sigma=1.2,
    ):
        reference_embedding, reference_latent = self.reference_encoder(
            x_ref=x_ref, key_padding_mask=x_ref_mask
        )

        prior_out = self.prior_encoder(
            phone_id=phone_id,
            duration=None,
            pitch=None,
            phone_mask=None,
            mask=None,
            ref_emb=reference_latent,
            ref_mask=x_ref_mask,
            is_inference=True,
        )

        condition_embedding = prior_out["prior_out"]

        bsz, l, _ = condition_embedding.shape
        if self.cfg.diffusion.diff_model_type == "Transofmer":
            z = (
                torch.randn(bsz, l, self.cfg.diffusion.diff_transformer.in_dim).to(
                    condition_embedding.device
                )
                / sigma
            )
        elif self.cfg.diffusion.diff_model_type == "WaveNet":
            z = (
                torch.randn(bsz, l, self.cfg.diffusion.diff_wavenet.input_size).to(
                    condition_embedding.device
                )
                / sigma
            )

        x0 = self.diffusion.reverse_diffusion(
            z=z,
            condition_embedding=condition_embedding,
            x_mask=None,
            reference_embedding=reference_embedding,
            n_timesteps=inference_steps,
        )

        return x0, prior_out

    @torch.no_grad()
    def reverse_diffusion_from_t(
        self,
        x,
        pitch=None,
        duration=None,
        phone_id=None,
        x_ref=None,
        phone_mask=None,
        x_mask=None,
        x_ref_mask=None,
        inference_steps=None,
        t=None,
    ):
        reference_embedding, reference_latent = self.reference_encoder(
            x_ref=x_ref, key_padding_mask=x_ref_mask
        )

        diffusion_step = (
            torch.ones(
                x.shape[0],
                dtype=x.dtype,
                device=x.device,
                requires_grad=False,
            )
            * t
        )
        diffusion_step = torch.clamp(diffusion_step, 1e-5, 1.0 - 1e-5)
        xt, _ = self.diffusion.forward_diffusion(x0=x, diffusion_step=diffusion_step)

        prior_out = self.prior_encoder(
            phone_id=phone_id,
            duration=duration,
            pitch=pitch,
            phone_mask=phone_mask,
            mask=x_mask,
            ref_emb=reference_latent,
            ref_mask=x_ref_mask,
            is_inference=True,
        )

        condition_embedding = prior_out["prior_out"]

        x0 = self.diffusion.reverse_diffusion_from_t(
            z=xt,
            condition_embedding=condition_embedding,
            x_mask=x_mask,
            reference_embedding=reference_embedding,
            n_timesteps=inference_steps,
            t_start=t,
        )

        return x0


def override_config(base_config, new_config):
    """Update new configurations in the original dict with the new dict

    Args:
        base_config (dict): original dict to be overridden
        new_config (dict): dict with new configurations

    Returns:
        dict: updated configuration dict
    """
    for k, v in new_config.items():
        if type(v) == dict:
            if k not in base_config.keys():
                base_config[k] = {}
            base_config[k] = override_config(base_config[k], v)
        else:
            base_config[k] = v
    return base_config


def get_lowercase_keys_config(cfg):
    """Change all keys in cfg to lower case

    Args:
        cfg (dict): dictionary that stores configurations

    Returns:
        dict: dictionary that stores configurations
    """
    updated_cfg = dict()
    for k, v in cfg.items():
        if type(v) == dict:
            v = get_lowercase_keys_config(v)
        updated_cfg[k.lower()] = v
    return updated_cfg


def _load_config(config_fn, lowercase=False):
    """Load configurations into a dictionary

    Args:
        config_fn (str): path to configuration file
        lowercase (bool, optional): whether changing keys to lower case. Defaults to False.

    Returns:
        dict: dictionary that stores configurations
    """
    with open(config_fn, "r") as f:
        data = f.read()
    config_ = json5.loads(data)
    if "base_config" in config_:
        # load configurations from new path
        p_config_path = os.path.join(os.getenv("WORK_DIR"), config_["base_config"])
        p_config_ = _load_config(p_config_path)
        config_ = override_config(p_config_, config_)
    if lowercase:
        # change keys in config_ to lower case
        config_ = get_lowercase_keys_config(config_)
    return config_


def load_config(config_fn, lowercase=False):
    """Load configurations into a dictionary

    Args:
        config_fn (str): path to configuration file
        lowercase (bool, optional): _description_. Defaults to False.

    Returns:
        JsonHParams: an object that stores configurations
    """
    config_ = _load_config(config_fn, lowercase=lowercase)
    # create an JsonHParams object with configuration dict
    cfg = JsonHParams(**config_)
    return cfg


def save_config(save_path, cfg):
    """Save configurations into a json file

    Args:
        save_path (str): path to save configurations
        cfg (dict): dictionary that stores configurations
    """
    with open(save_path, "w") as f:
        json5.dump(
            cfg, f, ensure_ascii=False, indent=4, quote_keys=True, sort_keys=True
        )





# def main():
#     model = UniAmphionTTS(cfg_path="configs/model/ns2_mel_wavenet.json")
#     model.load_state_dict(torch.load("/ckpts/ns2_mel_debug_wavenet/216k_step/pytorch_model.bin", map_location="cpu"))
#     print(model)

#     ref_wav, _ = librosa.load(..., sr=16000)
#     ref_wav = torch.from_numpy(ref_wav)
#     ref_wav = ref_wav[None, :]
#     ref_mel = mel_spectrogram(
#         ref_wav,
#         n_fft=1024,
#         num_mels=80,
#         sampling_rate=16000,
#         hop_size=200,
#         win_size=800,
#         fmin=0,
#         fmax=8000,
#     )

#     ref_mel = ref_mel.transpose(1, 2)
#     print(ref_mel.shape)

#     ref_mask = torch.ones(ref_mel.shape[0], ref_mel.shape[1])

#     reference_embedding, reference_latent = model.reference_encoder(
#         x_ref=ref_mel, key_padding_mask=ref_mask
#     )

#     print(reference_embedding.shape, reference_latent.shape)

# if __name__ == "__main__":
#     main()
