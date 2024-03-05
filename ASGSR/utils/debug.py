import librosa
import torchaudio
import numpy as np
import torch


def librosa_torchaudio(waveform_path):
    # torchaudio
    n_fft = 1024
    win_length = 800
    hop_length = 400
    stft_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=2.0)
    todb_transform = torchaudio.transforms.AmplitudeToDB()
    print('db_multiplier', todb_transform.db_multiplier)

    waveform, sample_rate = torchaudio.load(waveform_path)
    spectrogram = stft_transform(waveform)
    # print(torch.sum(spectrogram)) # tensor(638001.8750)
    spectrogram = todb_transform(spectrogram)  # -2878429.0
    print(torch.sum(spectrogram))

    # librosa
    waveform = waveform.numpy()
    X = librosa.stft(waveform, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    X = np.abs(X) ** 2
    # print(np.sum(X)) # 638002.0
    X = librosa.power_to_db(X, top_db=None)  # tensor(-4293940.)

    print(np.sum(X))

    X = torch.from_numpy(X)


if __name__ == '__main__':
    librosa_torchaudio('/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac/PA_T_0054000.flac')
