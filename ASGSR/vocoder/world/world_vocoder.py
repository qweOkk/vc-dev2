import pyworld
import librosa
import os
import numpy as np
import soundfile as sf
import tqdm

data_paths = [
    '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac',
    '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_dev/flac',
    '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_eval/flac'
]

data_lists = [
    '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.train.trn.txt',
    '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.dev.trl.txt',
    '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_cm_protocols/ASVspoof2019.PA.cm.eval.trl.txt'
]

save_path = '/mntnfs/lee_data1/wangli/ASGSR/ASVspoof2019/world'

for data_path, data_list in zip(data_paths, data_lists):
    with open(data_list, 'r') as f:
        for line in tqdm.tqdm(f.readlines()):
            line = line.strip()
            file_id = line.split(' ')[1]
            if line.split(' ')[4] == 'bonafide':
                file_path = os.path.join(data_path, file_id + '.flac')
                if os.path.exists(os.path.join(save_path, '{}.wav'.format(file_id))):
                    continue
                x, fs = librosa.load(file_path, sr=16000)
                x = x.astype(np.double)
                f0, sp, ap = pyworld.wav2world(x, fs)
                x_syn = pyworld.synthesize(f0, sp, ap, fs)
                sf.write(os.path.join(save_path, '{}.wav'.format(file_id)), x_syn, fs)
