# save all the used asvspoof audios

import os
import sys

sys.path.append('..')
from dataset.asvspoof2019 import ASVspoof2019

# config.data.ASVspoof2019.dataset.data_file = '/home/wangli/ASGSR/audio_record/enroll_eval_pairs.txt'
# config.data.ASVspoof2019.dataset.train_path = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac'
# config.data.ASVspoof2019.dataset.dev_path = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_dev/flac'
# config.data.ASVspoof2019.dataset.eval_path = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_eval/flac'
def save_all_asvspoof():
    asvspoof = ASVspoof2019(
        data_file='/home/wangli/ASGSR/audio_record/enroll_eval_pairs.txt',
        train_path='/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac',
        dev_path='/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_dev/flac',
        eval_path='/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_eval/flac',
    )
    for enroll_file_path, eval_file_path, _, _ in asvspoof._flist:
        os.system(f'cp {enroll_file_path} /mntnfs/lee_data1/wangli/ASGSR/all_asvspoof2019_used_wavs')
        os.system(f'cp {eval_file_path} /mntnfs/lee_data1/wangli/ASGSR/all_asvspoof2019_used_wavs')
        print(enroll_file_path, eval_file_path)
        # return

if __name__ == '__main__':
    save_all_asvspoof()