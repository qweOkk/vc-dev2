import argparse
import torch
import numpy as np
import torch
from tqdm import tqdm
from utils.util import load_config
import os
from models.tts.vc.vc_trainer import mel_spectrogram
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from models.sv.sv_model import sv_model
from models.tts.vc.ns2_uniamphion import mel_spectrogram

class SV_Dataset(Dataset):
    def __init__(self, data_dir, list_file):
        self.data_dir = data_dir
        self.list_file = list_file
        self.data, self.speaker2id = self._load_data()
        self.num_speakers = len(self.speaker2id.keys())
    
    def _load_data(self):
        data = []
        speaker2id = {}
        with open(self.list_file, "r") as f:
            for line in f:
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
            # pad by repeating the audio
            repetitions = (length // len(audio)) + 1
            padded_audio = np.tile(audio, repetitions)
            padded_audio = padded_audio[:length]  # Trim to desired length
            return padded_audio
        else:
            trimmed_audio = audio[:length]  # Trim to desired length
            return trimmed_audio
    
    def __getitem__(self, idx):
        speaker, audio_file = self.data[idx]
        speaker_id = self.speaker2id[speaker]
        audio, _ = librosa.load(audio_file, sr=16000) 
        audio = self._pad_or_trim(audio, length=3*16000)
        # to tensor
        audio = torch.tensor(audio).float()
        mask = torch.ones(len(audio) // 200)
        return {"speaker_id": speaker_id, "audio": audio, "mask": mask}
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        help="json files for configurations.",
        required=True,
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of dataloader workers."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Checkpoint for resume training or finetuning.",
    )
    parser.add_argument(
        "--cuda_id", 
        type=int, 
        default=5, 
        help="Cuda id for training."
    )
    args = parser.parse_args()

    data_dir = "/mnt/data3/share/voxceleb/voxceleb2"
    list_file = "/mnt/data3/share/voxceleb/vox2_train_list.txt"

    cfg = load_config(args.config)
    
    cuda_id = args.cuda_id
    device = torch.device(f"cuda:{cuda_id}")
    print("device", device)
    vc_model_path = args.checkpoint_path
    print("ckpt_path", vc_model_path)
    with device:
        torch.cuda.empty_cache()

    dataset = SV_Dataset(data_dir, list_file)
    train_loader = DataLoader(dataset, batch_size = 400, shuffle=True, num_workers=args.num_workers)
    num_speakers = dataset.num_speakers
    model = sv_model(cfg.model, num_speakers, vc_model_path)
    model = model.to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    num_epochs = 100
    best_epoch_loss = 100000
    best_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader):
            speaker_id = batch["speaker_id"]
            audio = batch["audio"]
            mask = batch["mask"]
            mel = mel_spectrogram(
                    audio,
                    n_fft=1024,
                    num_mels=80,
                    sampling_rate=16000,
                    hop_size=200,
                    win_size=800,
                    fmin=0,
                    fmax=8000,)  
            mel = mel.transpose(1, 2)
            mel = mel.to(device)
            mask = mask.to(device)
            speaker_id = speaker_id.to(device)
            loss = model(mel, mask, speaker_id)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        print(f"Epoch {epoch} Loss: {epoch_loss/len(train_loader)}")
        if epoch_loss < best_epoch_loss:
            best_epoch = epoch
            best_epoch_loss = epoch_loss
            best_epoch_loss = epoch_loss
            print(f"Best epoch: {best_epoch}")
            print(f"Best epoch loss: {best_epoch_loss}")
            print(f"Saving model at epoch {epoch}")
            torch.save(model.state_dict(), f"/mnt/data3/hehaorui/ckpt/sv/sv_model_best.pth")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f"/mnt/data3/hehaorui/ckpt/sv/sv_model_{epoch}.pth")
            print(f"Model saved at epoch {epoch}")
    torch.save(model.state_dict(), "sv_model_final.pth")


if __name__ == "__main__":
    main()