import os
import random

from utils.audio_utils import pad_cut
from torch.utils.data import Dataset
from utils.io_utils import load_waveform_torch
from utils.audio_utils import RandomSpeedChange
import os

os.path.join('.')
from dataset.augment import AugmentWAV


class VoxCeleb2Identification(Dataset):
    def __init__(self, root, meta_file, fix_sample_rate=16000, max_len=64000, augment=False, **kwargs):
        super().__init__()
        self._flist = self._get_flist(meta_file)
        self.root = root
        self.max_len = max_len
        self.fix_sample_rate = fix_sample_rate
        self.augment = augment
        if self.augment:
            self.augment_wav = AugmentWAV(musan_path=kwargs['musan_path'], rir_path=kwargs['rir_path'],
                                          max_len=self.max_len)
            self.speed_change = RandomSpeedChange(fix_sample_rate)

    def get_metadata(self, n):
        file_path = self._flist[n]  # id10003/na8-QEFmj44/00003.wav
        file_id = self._get_file_id(file_path)  # id10003-na8-QEFmj44-00003
        speaker_id = file_id.split("-")[0]  # id0003
        speaker_id = int(speaker_id[3:]) - 1  # 2
        return file_path, speaker_id, file_id

    def __getitem__(self, n):
        file_path, speaker_id, file_id = self.get_metadata(n)
        waveform, sr = load_waveform_torch(os.path.join(self.root, file_path))
        assert sr == self.fix_sample_rate, "sample rate must be {}, but {} {}".format(self.fix_sample_rate, file_path,
                                                                                      sr)
        waveform = pad_cut(waveform, self.max_len)
        if self.augment:
            waveform = self.augment_wav.process(waveform)
        return (waveform, speaker_id, file_id)

    def __len__(self):
        return len(self._flist)

    def _get_flist(self, meta_file):
        flist = []
        with open(meta_file, 'r') as f:
            for line in f:
                flist.append(line.strip().split(' ')[1])  # id00012/21Uxsk56VDQ/00001.wav
        return flist

    def _get_file_id(self, file_path):
        speaker_id, youtube_id, utterance_id = file_path.split("/")[-3:]
        utterance_id = utterance_id.split(".")[0]
        file_id = "-".join([speaker_id, youtube_id, utterance_id])
        return file_id


class Vox1Vox2Identification(Dataset):
    def __init__(self, vox1_root, vox2_root, vox1_meta_file, vox2_meta_file, fix_sample_rate=16000, max_len=64000,
                 augment=False, **kwargs):
        super().__init__()
        self._flist = self._get_flist(vox1_root, vox2_root, vox1_meta_file, vox2_meta_file)
        self.max_len = max_len
        self.fix_sample_rate = fix_sample_rate
        self.augment = augment
        if self.augment:
            self.augment_wav = AugmentWAV(musan_path=kwargs['musan_path'], rir_path=kwargs['rir_path'],
                                          max_len=self.max_len)
            self.speed_change = RandomSpeedChange(fix_sample_rate)
        self.spkid_dic = self._get_spkid_dic(vox1_meta_file, vox2_meta_file)

    def __getitem__(self, n):
        file_path = self._flist[n]
        speaker_id = file_path.split('/')[-3]
        speaker_id = self.spkid_dic[speaker_id]
        file_id = self._get_file_id(file_path)
        waveform, sr = load_waveform_torch(file_path)
        assert sr == self.fix_sample_rate, "sample rate must be {}, but {} {}".format(self.fix_sample_rate, file_path,
                                                                                      sr)

        if self.augment:
            waveform = self.speed_change(waveform)
        waveform = pad_cut(waveform, self.max_len)
        if self.augment:
            waveform = self.augment_wav.process(waveform)
        return (waveform, speaker_id, file_id)

    def __len__(self):
        return len(self._flist)

    def _get_flist(self, vox1_root, vox2_root, vox1_meta_file, vox2_meta_file):
        flist = []
        with open(vox1_meta_file, 'r') as f:
            for line in f:
                id, path = line.strip().split(' ')
                if id == '1':
                    flist.append(os.path.join(vox1_root, path))  # id00012/21Uxsk56VDQ/00001.wav
        with open(vox2_meta_file, 'r') as f:
            for line in f:
                flist.append(os.path.join(vox2_root, line.strip().split(' ')[1]))  # id00012/21Uxsk56VDQ/00001.wav
        return flist

    def _get_file_id(self, file_path):
        speaker_id, youtube_id, utterance_id = file_path.split("/")[-3:]
        utterance_id = utterance_id.split(".")[0]
        file_id = "-".join([speaker_id, youtube_id, utterance_id])
        return file_id

    def _get_spkid_dic(self, vox1_meta_file, vox2_meta_file):
        spkid = set()
        with open(vox1_meta_file, 'r') as f:
            for line in f:
                id, path = line.strip().split(' ')
                if id == '1':
                    spkid.add(path.split('/')[0])
        with open(vox2_meta_file, 'r') as f:
            for line in f:
                spkid.add(line.strip().split(' ')[0])
        spkid = sorted(spkid)
        spkid_dic = {}
        for i, spk in enumerate(spkid):
            spkid_dic[spk] = i
        return spkid_dic


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # # ------------------------vox2------------------------#
    # dataset = VoxCeleb2Identification(
    #     root='/mntcephfs/lab_data/wangli/voxceleb/voxceleb2',
    #     meta_file='/mntcephfs/lab_data/wangli/voxceleb/vox2_train_list.txt',
    #     augment=False,
    #     musan_path='/mntcephfs/lab_data/wangli/musan',
    #     rir_path='/mntcephfs/lab_data/wangli/rirs'
    # )
    #
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
    # print(dataset.__len__())
    # print(dataloader.__len__())
    #
    # for index, item in enumerate(dataloader):
    #     print(item[0].size())
    #     print(item[1])
    #     print(item[2])

    # ------------------------vox1+vox2------------------------#
    dataset = Vox1Vox2Identification(
        vox1_root='/mntcephfs/data/chenxi/datasets/VoxCeleb/VoxCeleb1/wav',
        vox2_root='/mntcephfs/lab_data/wangli/voxceleb/voxceleb2',
        vox1_meta_file='/mntcephfs/data/chenxi/datasets/VoxCeleb/VoxCeleb1/iden_split.txt',
        vox2_meta_file='/mntcephfs/lab_data/wangli/voxceleb/vox2_train_list.txt',
        augment=True,
        musan_path='/mntcephfs/lab_data/wangli/musan_split',
        rir_path='/mntcephfs/lab_data/wangli/RIRS_NOISES/simulated_rirs'
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
    print(dataset.__len__())
    print(dataloader.__len__())

    for index, item in enumerate(dataloader):
        print(item[0].size())
        print(item[1])
        print(item[2])
