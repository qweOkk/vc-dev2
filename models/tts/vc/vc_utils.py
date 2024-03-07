import torch
import pyworld as pw
import numpy as np
from librosa.filters import mel as librosa_mel_fn

mel_basis = {}
hann_window = {}
init_mel_and_hann = False

# online feature extraction
def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    global mel_basis, hann_window, init_mel_and_hann
    if not init_mel_and_hann:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
        init_mel_and_hann = True

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)
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