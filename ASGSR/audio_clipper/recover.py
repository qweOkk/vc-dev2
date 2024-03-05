import pickle
import json
import numpy as np
import torchaudio
import torchaudio.functional as F
import os
from pathlib import Path
import pdb
import torch
import math
# put this to the second at the beginning of the first pulse
# make it 0 if the audio's already starting with the first pulse
offset = int(
    # (19*3600*16000) + # hour
    # (36*60*16000) + # minute
    (21.2 * 16000) # second
)
# iphone, k40, mate
play_device = 'philip'
device = 'iphone'
part = '1'
# which folder to save the recovered audios
save_folder = f'/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/vox_bonafide/{play_device}/{device}/'
# os.makedirs(save_folder, exist_ok=True)

# the path to the full audio, must be ending with ".wav"
full_audio_path = f"/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/录音大师_{device}.{play_device}.wav"

# the path to the pkl file
pkl_path = f'/home/lijiaqi/phys_vocoder/audio_clipper/audio_meta_08700_1.pkl'

# config is done

# time after the pulse is 0.25s
extension_frames = int(0.25 * 16000)
def main():
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)

    # load the recorded audio
    # the recording should be edited out the start blank space
    rec_full_waveform, sr = torchaudio.load(full_audio_path)
    # rec_full_waveform = F.resample(rec_full_waveform, sr, 16000)
    assert sr == 16000
    if rec_full_waveform.shape[0] == 1:
        rec_full_waveform = rec_full_waveform[:, offset:]
    else:
        rec_full_waveform = rec_full_waveform[1:, offset:]



    curr_pulse_point_start = 0
    # find the largest pulse in the first 1s and reassign the start point
    # largest = 0
    # largest_idx = 0
    # for i in range(int(sr * 0.5)):
    #     # find the sample point with the largest amplitude
    #     if rec_full_waveform[0, i] > largest:
    #         largest = rec_full_waveform[0, i]
    #         largest_idx = i
    # curr_pulse_point_start = largest_idx
    # print(f'start pulse point: {curr_pulse_point_start}')

    # a delta matrix of the difference with each point's previous point
    delta_matrix = torch.zeros_like(rec_full_waveform)
    delta_matrix[:, 1:] = rec_full_waveform[:, 1:] - rec_full_waveform[:, :-1]

    curr_pulse_point_start -= extension_frames

    for entry in metadata:
        audio_path = entry['audio_path']


        time_duration = entry['time_duration']
        frame_duration = entry['sample_points']
        
        assert(sr == entry['sample_rate'])
        assert(sr == 16000)

        if audio_path != 'blank' and audio_path != 'pulse':
            audio_path = Path(audio_path)
            waveform_slice = rec_full_waveform[:, curr_pulse_point_start+extension_frames:curr_pulse_point_start+frame_duration+extension_frames]
            # print(curr_sample_point, curr_sample_point+frame_duration+extension_frames)
            # print(waveform_slice.shape)
            save_path = os.path.join(save_folder, f'{audio_path}')
            os.makedirs(str(Path(save_path).parent), exist_ok=True)
            torchaudio.save(save_path, waveform_slice, sr, bits_per_sample=16, encoding='PCM_S')
            print(f'write to {save_path}, duration: {time_duration:2f}s')

            # get the next pulse point in the 0.2s surrounding of the estimated pulse point
            estimated_pulse_point_start = curr_pulse_point_start+frame_duration+extension_frames+extension_frames
            # find the sample point with the largest amplitude
            largest = 0
            largest_idx = 0
            for i in range(estimated_pulse_point_start-int(sr*0.12), estimated_pulse_point_start+int(sr*0.12)):
                if rec_full_waveform[0, i] > largest:
                    largest = rec_full_waveform[0, i]
                    largest_idx = i
            curr_pulse_point_start = largest_idx
            print(f'diff: {curr_pulse_point_start - estimated_pulse_point_start}')
            print('estimate: ', end='')
            hours = math.floor((estimated_pulse_point_start+offset) / 16000 / 3600)
            minutes = math.floor((estimated_pulse_point_start+offset) / 16000 / 60) - hours * 60
            seconds = (estimated_pulse_point_start+offset) / 16000 - hours * 3600 - minutes * 60
            print(f'{hours} hours {minutes} minutes {seconds} seconds')
            print('curr: ', end='')
            hours = math.floor((curr_pulse_point_start+offset) / 16000 / 3600)
            minutes = math.floor((curr_pulse_point_start+offset) / 16000 / 60) - hours * 60
            seconds = (curr_pulse_point_start+offset) / 16000 - hours * 3600 - minutes * 60
            print(f'{hours} hours {minutes} minutes {seconds} seconds')
            try:
                assert(abs(curr_pulse_point_start - estimated_pulse_point_start) < 100)
            except:
                try:
                    smallest = 0
                    for i in range(estimated_pulse_point_start-int(sr*0.12), estimated_pulse_point_start+int(sr*0.12)):
                        if rec_full_waveform[0, i] < smallest:
                            smallest = rec_full_waveform[0, i]
                            smallest_idx = i
                    curr_pulse_point_start = smallest_idx
                    print(f'diff: {curr_pulse_point_start - estimated_pulse_point_start}')
                    print('estimate: ', end='')
                    hours = math.floor((estimated_pulse_point_start+offset) / 16000 / 3600)
                    minutes = math.floor((estimated_pulse_point_start+offset) / 16000 / 60) - hours * 60
                    seconds = (estimated_pulse_point_start+offset) / 16000 - hours * 3600 - minutes * 60
                    print(f'{hours} hours {minutes} minutes {seconds} seconds')
                    print('curr: ', end='')
                    hours = math.floor((curr_pulse_point_start+offset) / 16000 / 3600)
                    minutes = math.floor((curr_pulse_point_start+offset) / 16000 / 60) - hours * 60
                    seconds = (curr_pulse_point_start+offset) / 16000 - hours * 3600 - minutes * 60
                    print(f'{hours} hours {minutes} minutes {seconds} seconds')
                    assert(abs(curr_pulse_point_start - estimated_pulse_point_start) < 500)
                    
                except:
                    largest = 0
                    for i in range(estimated_pulse_point_start-int(sr*0.08), estimated_pulse_point_start+int(sr*0.08)):
                        if delta_matrix[0, i] > largest:
                            largest = delta_matrix[0, i]
                            largest_idx = i
                    curr_pulse_point_start = largest_idx
                    print(f'diff: {curr_pulse_point_start - estimated_pulse_point_start}')
                    print('estimate: ', end='')
                    hours = math.floor((estimated_pulse_point_start+offset) / 16000 / 3600)
                    minutes = math.floor((estimated_pulse_point_start+offset) / 16000 / 60) - hours * 60
                    seconds = (estimated_pulse_point_start+offset) / 16000 - hours * 3600 - minutes * 60
                    print(f'{hours} hours {minutes} minutes {seconds} seconds')
                    print('curr: ', end='')
                    hours = math.floor((curr_pulse_point_start+offset) / 16000 / 3600)
                    minutes = math.floor((curr_pulse_point_start+offset) / 16000 / 60) - hours * 60
                    seconds = (curr_pulse_point_start+offset) / 16000 - hours * 3600 - minutes * 60
                    print(f'{hours} hours {minutes} minutes {seconds} seconds')
                    try:
                        assert(abs(curr_pulse_point_start - estimated_pulse_point_start) < 1000)
                    except:
                        pdb.set_trace()
                # can `continue`, if check there's no problem with the output audio


main()