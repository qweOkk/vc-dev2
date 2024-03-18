from pathlib import Path
import torch
import torch.nn.functional as F
import joblib
import torch.nn as nn
import whisper
from transformers import WhisperModel
import warnings
import logging
from einops import rearrange, repeat

def noop(*args, **kwargs):
    pass

logging.root.setLevel(logging.ERROR)
warnings.warn = noop

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def round_down_nearest_multiple(num, divisor):
    return num // divisor * divisor


def curtail_to_multiple(t, mult, from_left=False):
    data_len = t.shape[-1]
    rounded_seq_len = round_down_nearest_multiple(data_len, mult)
    seq_slice = (
        slice(None, rounded_seq_len) if not from_left else slice(-rounded_seq_len, None)
    )
    return t[..., seq_slice]


class WhisperWithKmeans(nn.Module):
    """
    checkpoint and kmeans can be downloaded at https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
    or you can train your own
    """

    def __init__(
        self,
        whisper_path,
        kmeans_path,
    ):
        super().__init__()
        whisper_path = Path(whisper_path)
        kmeans_path = Path(kmeans_path)

        assert whisper_path.exists(), f"whisper path {whisper_path} does not exist"
        assert kmeans_path.exists(), f"kmeans path {kmeans_path} does not exist"


        self.model = WhisperModel.from_pretrained(whisper_path).encoder
        self.model.eval()

        kmeans = joblib.load(kmeans_path)

        self.kmeans = kmeans

        self.register_buffer(
            "cluster_centers", torch.from_numpy(kmeans.cluster_centers_)
        )

    @property
    def groups(self):
        return 1

    @property
    def codebook_size(self):
        return self.kmeans.n_clusters

    @property
    def downsample_factor(self):
        # todo: double check
        return 320
    
    def extract_whisper_input(self, audio, device):
        audio = torch.tensor(audio) #(B, T)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, device=device)
        return mel

    @torch.inference_mode()
    def forward(self, wav_input, flatten=True):
        whisper_mel = self.extract_whisper_input(wav_input, wav_input.device) #(B, 1500, 80)
        embed = self.model(whisper_mel).last_hidden_state #(B, 1500, 1024)
        embed = embed.permute((0, 2, 1))
        embed = F.interpolate(embed, scale_factor=8, mode="nearest")
        embed = F.interpolate(embed, scale_factor=0.2, mode="nearest")
        embed = embed.permute((0, 2, 1))

        batched_cluster_centers = repeat(
            self.cluster_centers, "c d -> b c d", b=embed.shape[0]
        )
        dists = -torch.cdist(embed, batched_cluster_centers, p=2)
        clusters = dists.argmax(dim=-1)
        quantize = F.embedding(clusters, self.cluster_centers)

        if flatten:
            return clusters, quantize

        return rearrange(clusters, "b ... -> b (...)"), quantize


def main():
    wav2vec = WhisperWithKmeans(
        whisper_path = '/mnt/data3/hehaorui/ckpt/whisper/whisper-base',
        kmeans_path ='/mnt/data4/hehaorui/whisper_kmeans/kmeans/libri-small-15s-percent_-1.0-clusters_1024.model'
    )
    wav2vec = wav2vec.to("cuda:5")

    with torch.no_grad():
        wav_input = torch.rand(2, 16000 * 4).to("cuda:5")
        print(wav_input.shape)
        code, quantize = wav2vec(wav_input)
        print(code, code.shape, quantize, quantize.shape)
        #torch.Size([2, 2400]) torch.Size([2, 2400, 512]) #B, T, D


if __name__ == "__main__":
    main()
