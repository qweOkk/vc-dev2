'''
VoxCeleb1 Identification Dataset:
    Number speakers:
        train:1251 dev:1251 test:1251


'''
import os
from pathlib import Path
from typing import List, Tuple, Union
from typing import Any, List, Optional

from torch import Tensor
from torch.utils.data import Dataset

from utils.io_utils import load_waveform_torch
from utils.audio_utils import pad_cut
from dataset.augment import AugmentWAV


class VoxCeleb1Identification(Dataset):
    def __init__(self, root: Union[str, Path], subset: str = "train", meta_file: str = '', fix_sample_rate=16000,
                 max_len=64000, augment=False,
                 **kwargs):
        '''
        Args:
            root: Path to the VoxCeleb1 dataset.
                eg:/mntcephfs/data/chenxi/datasets/VoxCeleb/VoxCeleb1/wav
            subset: train or dev or test, in iden_split.txt, 1 is train, 2 is dev, 3 is test.
            meta_file: identity meta file
                eg: /mntcephfs/data/chenxi/datasets/VoxCeleb/VoxCeleb1/iden_split.txt
                which contains 1 id10003/tCq2LcKO6xY/00002.wav
            fix_sample_rate: the fix sample rate
        '''
        super().__init__()
        if subset not in ["train", "dev", "test"]:
            raise ValueError("`subset` must be one of ['train', 'dev', 'test']")
        if not os.path.exists(meta_file):
            raise ValueError("meta_file {} not exists".format(meta_file))
        self._flist = self._get_flist(subset, meta_file)
        self.root = root
        self.subset = subset
        self.max_len = max_len
        self.fix_sample_rate = fix_sample_rate
        self.augment = augment
        if self.augment:
            self.augment_wav = AugmentWAV(musan_path=kwargs['musan_path'], rir_path=kwargs['rir_path'],
                                          max_len=self.max_len)

    def get_metadata(self, n: int) -> Tuple[str, int, int, str]:
        """Get metadata for the n-th sample from the dataset. Returns filepath instead of waveform,
        but otherwise returns the same fields as :py:func:`__getitem__`.

        Args:
            n (int): The index of the sample

        Returns:
            Tuple of the following items;

            str:
                Path to audio
            int:
                Speaker ID
            str:
                File ID
        """
        file_path = self._flist[n]  # id10003/na8-QEFmj44/00003.wav
        file_id = self._get_file_id(file_path)  # id10003-na8-QEFmj44-00003
        speaker_id = file_id.split("-")[0]  # id0003
        speaker_id = int(speaker_id[3:]) - 1  # 2
        return file_path, speaker_id, file_id

    def __getitem__(self, n: int):
        file_path, speaker_id, file_id = self.get_metadata(n)
        waveform, sr = load_waveform_torch(os.path.join(self.root, file_path))
        assert sr == self.fix_sample_rate, "sample rate must be {}, but {} {}".format(self.fix_sample_rate, file_path,
                                                                                      sr)
        if self.subset == 'train' and self.augment:
            waveform = pad_cut(waveform, self.max_len)
            waveform = self.augment_wav.process(waveform)
        return (waveform, speaker_id, file_id)

    def __len__(self) -> int:
        return len(self._flist)

    def _get_flist(self, subset, meta_file):
        # return [id10003/na8-QEFmj44/00003.wav,id10003/tCq2LcKO6xY/00002.wav, ...]
        f_list = []
        if subset == "train":
            index = 1
        elif subset == "dev":
            index = 2
        else:
            index = 3
        with open(meta_file, "r") as f:
            for line in f:
                id, path = line.split()
                if int(id) == index:
                    f_list.append(path)
        return sorted(f_list)

    def _get_file_id(self, file_path):
        speaker_id, youtube_id, utterance_id = file_path.split("/")[-3:]
        utterance_id = utterance_id.split(".")[0]
        file_id = "-".join([speaker_id, youtube_id, utterance_id])
        return file_id


class VoxCeleb1Verification(Dataset):
    def __init__(self, root: Union[str, Path], meta_file: str = '', fix_sample_rate=16000) -> None:
        super().__init__()
        self._flist = self._get_paired_flist(meta_file)
        self.root = root
        self.fix_sample_rate = fix_sample_rate

    def get_metadata(self, n: int):
        label, enroll_file, eval_file = self._flist[n]  # id10270/GWXujl-xAVM/00032.wav
        label = int(label)  # 0 means different speaker, 1 means same speaker
        return enroll_file, eval_file, label

    def __getitem__(self, n: int):
        enroll_file, eval_file, label = self.get_metadata(n)
        enroll_speaker = enroll_file.split("/")[0]
        eval_speaker = eval_file.split("/")[0]
        enroll_file_id = self._get_file_id(enroll_file)
        eval_file_id = self._get_file_id(eval_file)
        enroll_waveform, sr1 = load_waveform_torch(os.path.join(self.root, enroll_file))
        eval_waveform, sr2 = load_waveform_torch(os.path.join(self.root, eval_file))
        assert sr1 == self.fix_sample_rate, "sample rate must be {}, but {} {}".format(self.fix_sample_rate,
                                                                                       enroll_file, sr1)
        assert sr2 == self.fix_sample_rate, "sample rate must be {}, but {} {}".format(self.fix_sample_rate,
                                                                                       eval_file, sr2)
        return enroll_waveform, eval_waveform, 16000, label, enroll_file_id, eval_file_id, enroll_speaker, eval_speaker

    def __len__(self) -> int:
        return len(self._flist)

    def _get_paired_flist(self, veri_test_path: str):
        f_list = []
        with open(veri_test_path, "r") as f:
            for line in f:
                label, path1, path2 = line.split()
                f_list.append((label, path1, path2))
        return f_list

    def _get_file_id(self, file_path):
        speaker_id, youtube_id, utterance_id = file_path.split("/")[-3:]
        utterance_id = utterance_id.split(".")[0]
        file_id = "-".join([speaker_id, youtube_id, utterance_id])
        return file_id


class VoxCeleb1VerificationAttack(Dataset):
    '''
    The dataset after voxceleb1 verification attack.
    In this project, we only attack the evaluation audio rather than the enrollment audio.
    '''

    def __init__(self, attack_result_file, voxceleb1_file_dir, attack_file_dir):
        self._flist = []
        attack_result_file = open(attack_result_file)
        for line in attack_result_file.readlines():
            line = line.strip().split(' ')
            is_ori_success = line[2]
            is_ori_success = True if is_ori_success == 'True' else False

            enroll_file_id = line[0]  # id10270-x6uYqmx31kE-00001
            eval_file_id = line[1]  # id10270-x6uYqmx31kE-00003_id10273-8cfyJEV7hP8-00004

            enroll_spk_id = enroll_file_id[:7]  # id10270
            eval_spk_id = eval_file_id[25:32]  # id10273
            ori_label = int(line[3])

            enroll_file_path = os.path.join(
                voxceleb1_file_dir,
                enroll_file_id[:7],  # id10270
                enroll_file_id[8:8 + 11],  # x6uYqmx31kE
                enroll_file_id[8 + 11 + 1:] + '.wav'  # 00001.wav
            )
            eval_file_path = os.path.join(attack_file_dir, eval_file_id + '.wav')

            self._flist.append([enroll_file_path, eval_file_path, enroll_file_id, eval_file_id, ori_label, is_ori_success])
        attack_result_file.close()

    def __getitem__(self, i):
        enroll_file_path, eval_file_path, enroll_file_id, eval_file_id, ori_label, is_ori_success = self._flist[i]
        enroll_waveform, _ = load_waveform_torch(enroll_file_path)
        eval_waveform, _ = load_waveform_torch(eval_file_path)
        return enroll_waveform, eval_waveform, enroll_file_id, eval_file_id, ori_label, is_ori_success

    def __len__(self):
        return len(self._flist)


def stat_vox1_info(veri_file='/mntcephfs/data/chenxi/datasets/VoxCeleb/VoxCeleb1/veri_test.txt'):
    import matplotlib.pyplot as plt
    veri_file = open(veri_file)
    spk_dict = {}
    number_same = 0
    number_diff = 0
    spk_dict_same = {}
    spk_dict_diff = {}
    for line in veri_file.readlines():
        line = line.strip().split(' ')
        spk1 = line[1].split('/')[0]
        spk2 = line[2].split('/')[0]
        spk = '{}_{}'.format(spk1, spk2)
        label = line[0]
        if label == '1':
            number_same += 1
            if spk not in spk_dict_same:
                spk_dict_same[spk] = 1
            else:
                spk_dict_same[spk] += 1
        else:
            number_diff += 1
            if spk not in spk_dict_diff:
                spk_dict_diff[spk] = 1
            else:
                spk_dict_diff[spk] += 1

    values = list(spk_dict_same.values())
    plt.hist(values)
    plt.title('Same speaker sample pair statistics (total: {}, min:{}, max:{})'.format(number_same, min(values), max(values)))
    plt.xlabel('Count')
    # spk_dict = sorted(spk_dict.items(), key=lambda x: x[1], reverse=True)
    plt.savefig('same_speaker_pair_stat.png')
    plt.close()

    values = list(spk_dict_diff.values())
    plt.hist(values)
    plt.title('Diff speaker sample pair statistics (total: {}, min:{}, max:{})'.format(number_diff, min(values), max(values)))
    plt.xlabel('Count')
    # spk_dict = sorted(spk_dict.items(), key=lambda x: x[1], reverse=True)
    plt.savefig('diff_speaker_pair_stat.png')
    plt.close()


if __name__ == '__main__':
    # from torch.utils.data import DataLoader
    #
    # voxceleb1_dataset = VoxCeleb1Identification(
    #     root='/mntcephfs/data/chenxi/datasets/VoxCeleb/VoxCeleb1/wav',
    #     subset='train',
    #     meta_file='/mntcephfs/data/chenxi/datasets/VoxCeleb/VoxCeleb1/iden_split.txt',
    #     fix_sample_rate=16000
    # )
    # voxceleb1_dataloader = DataLoader(voxceleb1_dataset, batch_size=64, num_workers=1)
    # speaker_id = []
    # print(voxceleb1_dataset.__len__())
    # print(voxceleb1_dataloader.__len__())
    #
    # for index, item in enumerate(voxceleb1_dataloader):
    #     print(item[0].size())
    #     print(item[1])
    #     print(item[2])

    from torch.utils.data import DataLoader
    dataset = VoxCeleb1Verification(
        root='/mntcephfs/data/chenxi/datasets/VoxCeleb/VoxCeleb1/wav',
        meta_file='/home/wangli/ASGSR/utils/vox1_uniform_sample_25.txt',
    )
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1)
    for index, item in enumerate(dataloader):
        print(item[0].size())
        print(item[1].size())
        print(item[2])
        print(item[3])
        print(item[4])
        print(item[5])
        break

