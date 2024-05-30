import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, WavLMForSequenceClassification, TrainingArguments, Trainer
import librosa

# Dataset class
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
            lines = f.readlines()
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
        return {"input_values": audio, "labels": torch.tensor(speaker_id)}



# Define training arguments
training_args = TrainingArguments(
    output_dir="./wavlm_speaker_classification_results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir='/mnt/data3/hehaorui/ckpt/sv/logs',
    logging_steps=10,
)

data_dir = "/mnt/data3/share/voxceleb/voxceleb2"
list_file = "/mnt/data3/share/voxceleb/vox2_train_list.txt"

train_dataset = SV_Dataset(data_dir, list_file, sample = 10000)

# Loading the processor and the model
processor = Wav2Vec2Processor.from_pretrained("/mnt/data3/hehaorui/ckpt/wavlm/wavlm-base")
model = WavLMForSequenceClassification.from_pretrained("/mnt/data3/hehaorui/ckpt/wavlm/wavlm-base", num_labels=len(train_dataset.num_speakers))
 
# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    tokenizer=processor.feature_extractor,
)

# Start training
trainer.train()

# Save the final model
model.save_pretrained("/mnt/data3/hehaorui/ckpt/sv/final_speaker_model")
