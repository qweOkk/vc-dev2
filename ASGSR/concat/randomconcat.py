import os
import random
import torchaudio
import torch
import json
import tqdm

def generate_data(audio_folder, output_folder, num_data):
    audio_files = os.listdir(audio_folder)
    audio_list = []
    for file in audio_files:
        if file.endswith('.wav') or file.endswith('.mp3'):
            audio_list.append(os.path.join(audio_folder, file))

    data_list = []
    for i in tqdm.tqdm(range(num_data)):
        audio1 = random.choice(audio_list)
        audio2 = random.choice(audio_list)

        # Load audio files
        waveform1, sample_rate1 = torchaudio.load(audio1)
        waveform2, sample_rate2 = torchaudio.load(audio2)

        # Randomly select duration from waveform1
        duration = random.uniform(0.5, 2.0) * sample_rate1
        if duration > waveform1.size(1):
            duration = waveform1.size(1)
        clip_start_index = random.randint(0, waveform1.size(1) - int(duration))
        clip_end_index = clip_start_index + int(duration)

        # Randomly select insert index from waveform2
        insert_start_index = random.randint(0, waveform2.size(1))
        insert_end_index = insert_start_index + int(duration)

        # Concatenate waveform1 and waveform2
        combined_waveform = torch.cat([waveform2[:, :insert_start_index], waveform1[:, clip_start_index:clip_end_index], waveform2[:, insert_start_index:]], dim=1)

        # Record data
        data = {
            'concat_id': os.path.splitext(os.path.basename(audio2))[0],
            'concat_start_time': insert_start_index / sample_rate1,
            'concat_end_time': insert_end_index / sample_rate1,
            'concat_clip_start_index': insert_start_index,
            'concat_clip_end_index': insert_end_index,
            'truncation_id': os.path.splitext(os.path.basename(audio1))[0],
            'random_start_time': clip_start_index / sample_rate1,
            'random_end_time': clip_end_index / sample_rate1,
            'random_clip_start_index': clip_start_index,
            'random_clip_end_index': clip_end_index,
            'duration': duration / sample_rate1
        }
        data_list.append(data)

        # Save combined waveform to output folder
        output_filename = '{}-{}.wav'.format(data['concat_id'], data['truncation_id'])
        output_path = os.path.join(output_folder, output_filename)
        torchaudio.save(output_path, combined_waveform, sample_rate1)

        # Save data_list to file
        data_file = os.path.join(output_folder, 'data.json')
        with open(data_file, 'w') as f:
            json.dump(data_list, f)

    return data_list

audio_folder = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train_bonafide'
output_folder = '/mntcephfs/lab_data/wangli/concat'
num_data = 10000

generated_data = generate_data(audio_folder, output_folder, num_data)
