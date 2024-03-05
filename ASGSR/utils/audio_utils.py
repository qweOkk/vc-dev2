import torch
import random
import torchaudio


def pad_cut(sig, max_len=64000):
    sig_len = sig.size()[1]
    if sig_len == max_len:
        return sig
    elif sig_len < max_len:
        padded_sig = torch.zeros(1, max_len)
        padded_sig[:, :sig_len] = sig
        return padded_sig
    else:
        start_index = random.randint(0, sig_len - max_len)
        cutted_sig = sig[:, start_index:start_index + max_len]
        return cutted_sig


class RandomSpeedChange:
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def __call__(self, audio_data):
        speed_factor = random.choice([0.9, 1.0, 1.1])
        if speed_factor == 1.0:  # no change
            return audio_data

        # change speed and resample to original rate:
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_data, self.sample_rate, sox_effects)
        return transformed_audio
