# statistic the duration distribution of the audio files in the dataset
import os
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_audio_duration(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    duration = waveform.shape[1] / sample_rate
    return duration

def plot_and_save_duration_distribution(audio_paths, save_path, title):
    durations = [get_audio_duration(path) for path in audio_paths]
    plt.hist(durations, bins=20, edgecolor='black')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Count')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def process_folder(folder_path, save_path, title):
    audio_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                audio_paths.append(os.path.join(root, file))
    plot_and_save_duration_distribution(audio_paths, save_path, title)

# Process individual folders
folder_paths = [
    '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/wav',
    '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_eval/wav',
    '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_dev/wav'
]
titles = [
    'ASVspoof2019_PA_train',
    'ASVspoof2019_PA_eval',
    'ASVspoof2019_PA_dev'
]

for folder_path, title in tqdm(zip(folder_paths, titles), total=len(folder_paths), desc='Processing folders'):
    save_path = f'duration_distribution_{title}.png'
    process_folder(folder_path, save_path, title)
    print(f"Duration distribution for {title} saved at {save_path}")

# Process all folders and calculate overall duration distribution
all_audio_paths = []
for folder_path in folder_paths:
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.wav'):
                all_audio_paths.append(os.path.join(root, file))

save_path = 'duration_distribution_total.png'
plot_and_save_duration_distribution(all_audio_paths, save_path, 'Overall Duration Distribution')
print(f"Overall duration distribution saved at {save_path}")
