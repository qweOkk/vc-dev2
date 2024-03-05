from torch.utils.data import Dataset
import librosa
import os
import torch



class SVDataset(Dataset):
    def __init__(self, data, folder_path):
        self.data = data
        self.folder_path = folder_path

    def __len__(self):
        return len(self.data)

    def load_wav(self,ref_wav_path):
        ref_wav,_ = librosa.load(ref_wav_path, sr=16000) 
        ref_wav = librosa.util.fix_length(ref_wav, size=16000 * 15)
        speech = torch.tensor(ref_wav, dtype=torch.float32)
        pad_size = 1600 - speech.shape[0] % 1600
        speech = torch.nn.functional.pad(speech, (0, pad_size))
        return speech

    def __getitem__(self, idx):
        label = int(self.data['Label'].iloc[idx])
        first_file = self.data['First'].iloc[idx]
        second_file = self.data['Second'].iloc[idx]
        # Check if first_file is an absolute path, if not, join with folder_path
        if not os.path.isabs(first_file):
            first_file = os.path.join(self.folder_path, first_file)
        # Check if second_file is an absolute path, if not, join with folder_path
        if not os.path.isabs(second_file):
            second_file = os.path.join(self.folder_path, second_file)
        # Load and process audio data
        first_wav = self.load_wav(first_file)
        second_wav = self.load_wav(second_file)
        return first_wav, second_wav, label
    