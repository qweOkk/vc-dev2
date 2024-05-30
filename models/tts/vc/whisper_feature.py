import torch
import random
import whisper
from whisper.utils import exact_div
import torch.nn as nn
import torchvision
from transformers import WhisperModel
import torch.nn.functional as F

class WhisperNormal(nn.Module):
    def __init__(
        self,
        whisper_path = "/mnt/data3/hehaorui/pretrained_models/whisper/whisper-base",
    ):
        super().__init__()
        self.whisper_encoder = WhisperModel.from_pretrained(whisper_path).encoder.eval()

    def extract_whisper_input(self, audio, inference=False):
        real_num_tokens = exact_div(len(audio[0]), 320) # n_samples % n_samples per token = num_tokens
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio, device=audio.device)
        if inference:
            mel = self.spec_augment(mel, 72)
        else:
            mel = self.spec_augment(mel, random.randint(68, 92)) #扰动
        return mel, real_num_tokens
    
    def scale_to_target_hop_size(self, embed):
        # whisper 320 --> 200
        embed = embed.permute((0, 2, 1)) #BTD -> BDT
        embed = F.interpolate(embed, scale_factor=8, mode="nearest")
        embed = F.interpolate(embed, scale_factor=0.2, mode="nearest")
        embed = embed.permute((0, 2, 1)) 
        return embed
    
    @torch.inference_mode()
    def forward(self, wav_input, inference=False):
        self.whisper_encoder.to(wav_input.device)
        with torch.no_grad():
            whisper_mel, real_num_tokens = self.extract_whisper_input(wav_input, inference) 
            embed = self.whisper_encoder(whisper_mel).last_hidden_state 
            embed= embed[:, :real_num_tokens, :] 
            embed = self.scale_to_target_hop_size(embed)
        return None, embed

    def spec_augment(self, mel, height):
        """
        Args:
            mel: tensor (..., n_mels, frames)
            height: int 68-92 for default 80 mels
        """
        tgt = torchvision.transforms.functional.resize(mel, (height, mel.shape[-1]))
        if height >= mel.shape[-2]:
            return tgt[:, :mel.shape[-2], :]
        else:
            silence = tgt[:, -1:, :].repeat(1, mel.shape[-2] - height, 1)
            silence += torch.randn_like(silence) / 10
            return torch.cat((tgt, silence), 1)
        
# main
if __name__ == "__main__":
    model = WhisperNormal()
    audio = torch.randn(1, 16000)
    _,emb = model(audio)
    print(emb.shape)
 