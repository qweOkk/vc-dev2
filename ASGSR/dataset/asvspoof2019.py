import os.path
from torch.utils.data import Dataset
from utils.io_utils import load_waveform_torch


class ASVspoof2019(Dataset):
    def __init__(self, data_file, train_path, dev_path, eval_path):

        self._flist = []
        self.train_path = train_path
        self.dev_path = dev_path
        self.eval_path = eval_path

        # self.fnamepair_to_idx = dict()  # {enroll_fname}_{eval_fname}

        with open(data_file, 'r') as f:
            for line in f:
                enroll_speaker, enroll_file, eval_speaker, eval_file, label = line.strip().split(' ')[:5]
                enroll_file_path = self.full_path(enroll_file)
                eval_file_path = self.full_path(eval_file)
                # self.fnamepair_to_idx[f'{enroll_file}_{eval_file}'] = len(self._flist)
                self._flist.append([enroll_file_path, eval_file_path, enroll_file, eval_file, int(label), enroll_speaker, eval_speaker])

    def __len__(self):
        return len(self._flist)

    def __getitem__(self, i):
        enroll_file_path, eval_file_path, enroll_file, eval_file, label, enroll_speaker, eval_speaker = self._flist[i]
        enroll_waveform, sr = load_waveform_torch(enroll_file_path)
        assert sr == 16000, 'sample rate of enrollment audio file should be 16000 but {}'.format(sr)
        eval_waveform, sr = load_waveform_torch(eval_file_path)
        assert sr == 16000, 'sample rate of evaluation audio file should be 16000 but {}'.format(sr)

        return enroll_waveform, eval_waveform, 16000, label, enroll_file, eval_file, enroll_speaker, eval_speaker

    def full_path(self, file):
        if file.split('_')[1] == 'T':  # train
            file_path = os.path.join(self.train_path, file + '.flac')
        elif file.split('_')[1] == 'D':  # dev
            file_path = os.path.join(self.dev_path, file + '.flac')
        elif file.split('_')[1] == 'E':  # eval
            file_path = os.path.join(self.eval_path, file + '.flac')
        else:
            raise ValueError('Unknown file type: {}'.format(file))
        return file_path


class ASVspoof2019TransferAttack(Dataset):
    def __init__(self, attack_result_file, attack_file_dir, asvspoof_train_path, asvspoof_dev_path, asvspoof_eval_path):
        self.asvspoof_train_path = asvspoof_train_path
        self.asvspoof_dev_path = asvspoof_dev_path
        self.asvspoof_eval_path = asvspoof_eval_path
        self.attack_result_file = attack_result_file
        self.attack_file_dir = attack_file_dir
        self._flist = []
        with open(attack_result_file, 'r') as f:
            for line in f:
                enroll_file, adversarial_file, is_success = line.strip().split(' ')
                self._flist.append([enroll_file, adversarial_file, is_success])

    def __getitem__(self, i):
        enroll_file, adversarial_file, is_success = self._flist[i]
        enroll_file_path = self.full_path(enroll_file)
        adversarial_file_path = os.path.join(self.attack_file_dir, adversarial_file + '.wav')
        enroll_waveform, sr = load_waveform_torch(enroll_file_path)
        assert sr == 16000, 'sample rate of enrollment audio file should be 16000 but {}'.format(sr)
        adversarial_waveform, sr = load_waveform_torch(adversarial_file_path)
        assert sr == 16000, 'sample rate of evaluation audio file should be 16000 but {}'.format(sr)
        return enroll_waveform, adversarial_waveform, 16000, is_success

    def full_path(self, file):
        if file.split('_')[1] == 'T':  # train
            file_path = os.path.join(self.asvspoof_train_path, file + '.flac')
        elif file.split('_')[1] == 'D':  # dev
            file_path = os.path.join(self.asvspoof_dev_path, file + '.flac')
        elif file.split('_')[1] == 'E':  # eval
            file_path = os.path.join(self.asvspoof_eval_path, file + '.flac')
        else:
            raise ValueError('Unknown file type: {}'.format(file))
        return file_path
