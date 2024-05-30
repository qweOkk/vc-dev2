import numpy as np
import torch
import random
import os
import librosa
import soundfile as sf
import tqdm
import rir_generator as rir

SAMPLE_RATE=16000
# get all wav in "/mnt/petrelfs/hehaorui/data/datasets/noise/noise_test"
noise_test_dir = "/home/hehaorui/code/Amphion/MS-SNSD/noise_test"
noise_filenames = []
for root, dirs, files in os.walk(noise_test_dir):
    for file in files:
        if file.endswith(".wav"):
            noise_filenames.append(os.path.join(root, file))

def snr_mixer(clean, noise, snr):
    # Normalizing to -25 dB FS
    rmsclean = (clean**2).mean()**0.5
    epsilon = 1e-10
    rmsclean = max(rmsclean, epsilon)
    scalarclean = 10 ** (-25 / 20) / rmsclean
    clean = clean * scalarclean

    rmsnoise = (noise**2).mean()**0.5
    rmsnoise = max(rmsnoise, epsilon)
    if rmsnoise == epsilon:
        return clean / scalarclean
    scalarnoise = 10 ** (-25 / 20) /rmsnoise
    noise = noise * scalarnoise
    rmsnoise = (noise**2).mean()**0.5
    
    # Set the noise level for a given SNR
    noisescalar = np.sqrt(rmsclean / (10**(snr/20)) / rmsnoise)
    noisenewlevel = noise * noisescalar
    noisyspeech = clean + noisenewlevel
    noisyspeech_tensor = torch.tensor(noisyspeech, dtype=torch.float32)
    return noisyspeech_tensor

def add_noise(clean):
    # self.noise_filenames: list of noise files
    random_idx = np.random.randint(0, np.size(noise_filenames))
    selected_noise_file = noise_filenames[random_idx]
    noise, _ = librosa.load(selected_noise_file, sr=SAMPLE_RATE)
    clean = clean.cpu().numpy()
    if len(noise)>=len(clean):
        noise = noise[0:len(clean)] #截取噪声的长度
    else:
        while len(noise)<=len(clean): #如果噪声的长度小于语音的长度
            random_idx = (random_idx + 1)%len(noise_filenames) #随机读一个噪声
            newnoise, fs = librosa.load(selected_noise_file, sr=SAMPLE_RATE)
            noiseconcat = np.append(noise, np.zeros(int(fs * 0.2)))#在噪声后面加上0.2静音
            noise = np.append(noiseconcat, newnoise)#拼接噪声
    noise = noise[0:len(clean)] #截取噪声的长度
    #随机sample一个小于20大于0的随机数
    snr = random.uniform(0.0,5.0) #随机选择SNR级别
    noisyspeech = snr_mixer(clean=clean, noise=noise, snr=snr) #根据随机的SNR级别，混合生成带噪音频
    del noise
    return noisyspeech


RAW_VCTK_DIR_1 = "/mnt/data2/hehaorui/datasets/VCTK/prompt"
RAW_VCTK_DIR_2 = "/mnt/data2/hehaorui/datasets/VCTK/source"
# get_all_clean speeches

clean_files = []
for root, dirs, files in os.walk(RAW_VCTK_DIR_1):
    for file in files:
        if file.endswith(".wav"):
            clean_files.append(os.path.join(root, file))

for root, dirs, files in os.walk(RAW_VCTK_DIR_2):
    for file in files:
        if file.endswith(".wav"):
            clean_files.append(os.path.join(root, file))
    
#add_noise and 50% add_reverb randomly, save to RAW_VCTK_DIR_1+"noisy" and  RAW_VCTK_DIR_2+"noisy"


# Create the directories for saving the noisy files if they don't already exist
noisy_dir_1 = RAW_VCTK_DIR_1+ "noisy2"
noisy_dir_2 = RAW_VCTK_DIR_2+ "noisy2"

os.makedirs(noisy_dir_1, exist_ok=True)
os.makedirs(noisy_dir_2, exist_ok=True)

# Function to save the audio file
def save_audio(audio_tensor, sample_rate, file_path):
    audio_numpy = audio_tensor.numpy()
    sf.write(file_path, audio_numpy, sample_rate)

# Process each clean file
for clean_file in tqdm.tqdm(clean_files):
    # Load the clean speech
    clean_speech, _ = librosa.load(clean_file, sr=SAMPLE_RATE)
    clean_speech_tensor = torch.tensor(clean_speech, dtype=torch.float32)
    
    # Add noise to the clean speech
    noisy_speech_tensor = add_noise(clean_speech_tensor)
    assert noisy_speech_tensor.shape == clean_speech_tensor.shape, print("add", noisy_speech_tensor.shape, clean_speech_tensor.shape)
    # With 50% chance, add reverb

    # Determine the save path based on the original file directory
    if RAW_VCTK_DIR_1 in clean_file:
        save_path = clean_file.replace(RAW_VCTK_DIR_1, noisy_dir_1)
    else:
        save_path = clean_file.replace(RAW_VCTK_DIR_2, noisy_dir_2)
    # save_path = clean_file.replace(RAW_VCTK_DIR_2, noisy_dir_2)
    # Save the noisy (and possibly reverberated) audio
    print(save_path)
    save_audio(noisy_speech_tensor, SAMPLE_RATE, save_path)

print("Processing complete.")

