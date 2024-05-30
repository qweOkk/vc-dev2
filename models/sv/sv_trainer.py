import argparse
import torch
import numpy as np
import os
import random
from tqdm import tqdm
from utils.util import load_config
from models.tts.vc.vc_trainer import mel_spectrogram  # Ensure no duplicate import
from models.sv.sv_model import SVMODEL,SVMODEL_SSL
import librosa
from torch.utils.data import Dataset, DataLoader

class SV_Dataset(Dataset):
    def __init__(self, data_dir, list_file):
        self.data_dir = data_dir
        self.list_file = list_file
        self.data, self.speaker2id = self._load_data()
        self.num_speakers = len(self.speaker2id)
        print(f"Number of speakers: {self.num_speakers}")

    def _load_data(self):
        data = []
        speaker2id = {}
        with open(self.list_file, "r") as f:
            lines = f.readlines()  # 读取所有行到一个列表中

        # 如果数据行数多于 10000 行，随机选择 10000 行
        if len(lines) > 10000:
            lines = random.sample(lines, 10000)

        for line in lines:
            line = line.strip().split()
            speaker = line[0]
            audio = os.path.join(self.data_dir, line[1])
            if speaker not in speaker2id:
                speaker2id[speaker] = len(speaker2id)
            data.append((speaker, audio))

        return data, speaker2id

    def __len__(self):
        return len(self.data)

    def _pad_or_trim(self, audio, length=5*16000):
        if len(audio) < length:
            padded_audio = np.pad(audio, (0, length - len(audio)), mode='wrap')
            return padded_audio
        else:
            trimmed_audio = audio[:length]
            return trimmed_audio

    def __getitem__(self, idx):
        speaker, audio_file = self.data[idx]
        speaker_id = self.speaker2id[speaker]
        audio, _ = librosa.load(audio_file, sr=16000) 
        audio = self._pad_or_trim(audio, length=3*16000)
        audio = torch.tensor(audio).float()
        return {"speaker_id": speaker_id, "audio": audio}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json", help="json files for configurations.", required=True)
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint for resume training or finetuning.")
    parser.add_argument("--cuda_id", type=int, default=5, help="Cuda id for training.")
    parser.add_argument("--ssl", type=str, default='wav2vec', help="SSL model to use for speaker verification.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    data_dir = "/mnt/data3/share/voxceleb/voxceleb2"
    list_file = "/mnt/data3/share/voxceleb/vox2_train_list.txt"

    dataset = SV_Dataset(data_dir, list_file)
    
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=args.num_workers)
    num_speakers = dataset.num_speakers
    if args.ssl == 'VC':
        print("Using VC model as the SSL model.")
        model = SVMODEL(cfg['model'], num_speakers, args.checkpoint_path).to(device)
    else:
        print("Using wav2vec 2.0 as the SSL model.")
        model = SVMODEL_SSL(num_speakers).to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    num_epochs = 100
    best_epoch_loss = float('inf')
    best_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            speaker_id = batch["speaker_id"].to(device)
            audio = batch["audio"].to(device)
            mask = batch["mask"].to(device)
            if args.ssl == 'VC':
                mel = mel_spectrogram(audio, n_fft=1024, num_mels=80, sampling_rate=16000, hop_size=200, win_size=800, fmin=0, fmax=8000)
                mel = mel.transpose(1, 2)
                loss = model(mel, mask, speaker_id)
            else:
                loss = model(audio, mask, speaker_id)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch} Loss: {epoch_loss}")
        if epoch_loss < best_epoch_loss:
            best_epoch = epoch
            best_epoch_loss = epoch_loss
            print(f"New best epoch: {best_epoch} with loss: {best_epoch_loss}")
            torch.save(model.state_dict(), f"/mnt/data3/hehaorui/ckpt/sv/sv_model_best.pth")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"/mnt/data3/hehaorui/ckpt/sv/sv_model_{epoch}.pth")

    torch.save(model.state_dict(), "sv_model_final.pth")

if __name__ == "__main__":
    main()
