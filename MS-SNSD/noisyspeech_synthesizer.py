"""
@author: chkarada
"""
import glob
import numpy as np
import soundfile as sf
import os
import argparse
import configparser as CP
from audiolib import audioread, audiowrite, snr_mixer
from tqdm import tqdm
from collections import defaultdict
import json

def main(cfg):
    speaker_tracker_save_path = "speakers_stats_test.json" # added a json to keep track of speaker stats
    noise_tracker_save_path = "noise_stats_test.json" # added a json to keep track of noise stats
    seed_value = 42 # set a random seed for reproduceability
    np.random.seed(seed_value)

    snr_lower = float(cfg["snr_lower"]) # lower bound of snr
    snr_upper = float(cfg["snr_upper"]) # upper bound of snr
    total_snrlevels = float(cfg["total_snrlevels"]) # total number of snr levels
    
    clean_dir = os.path.join(os.path.dirname(__file__), 'clean_test')
    if cfg["speech_dir"]!='None':
        clean_dir = cfg["speech_dir"]
    if not os.path.exists(clean_dir):
        assert False, ("Clean speech data is required")
    
    noise_dir = os.path.join(os.path.dirname(__file__), 'noise_test')
    if cfg["noise_dir"]!='None':
        noise_dir = cfg["noise_dir"]
    if not os.path.exists(noise_dir):
        assert False, ("Noise data is required")
        
    fs = float(cfg["sampling_rate"])
    audioformat = cfg["audioformat"]
    total_hours = float(cfg["total_hours"])
    audio_length = float(cfg["audio_length"])
    silence_length = float(cfg["silence_length"])
    noisyspeech_dir = os.path.join(os.path.dirname(__file__), '/nvme/uniamphion/se/Libritts_SE/wavs/test/', 'NoisySpeech_testing')
    if not os.path.exists(noisyspeech_dir):
        os.makedirs(noisyspeech_dir)
    clean_proc_dir = os.path.join(os.path.dirname(__file__), '/nvme/uniamphion/se/Libritts_SE/wavs/test/', 'CleanSpeech_testing')
    if not os.path.exists(clean_proc_dir):
        os.makedirs(clean_proc_dir)
    noise_proc_dir = os.path.join(os.path.dirname(__file__), '/nvme/uniamphion/se/Libritts_SE/wavs/test/', 'Noise_testing')
    if not os.path.exists(noise_proc_dir):
        os.makedirs(noise_proc_dir)
        
    total_secs = total_hours*60*60
    total_samples = int(total_secs * fs)
    audio_length = int(audio_length*fs)
    SNR = np.linspace(int(snr_lower), int(snr_upper), int(total_snrlevels))
    cleanfilenames = glob.glob(os.path.join(clean_dir, audioformat)) #
    if cfg["noise_types_excluded"]=='None':
        noisefilenames = glob.glob(os.path.join(noise_dir, audioformat))
    else:
        filestoexclude = cfg["noise_types_excluded"].split(',')
        noisefilenames = glob.glob(os.path.join(noise_dir, audioformat))
        for i in range(len(filestoexclude)):
            noisefilenames = [fn for fn in noisefilenames if not os.path.basename(fn).startswith(filestoexclude[i])]
    
    filecounter = 0
    num_samples = 0
    prev_num_samples = 0 # for progress bar
    speakers_tracker = defaultdict(int) # only use for libritts
    noise_tracker = defaultdict(int)
    progress_bar = tqdm(total=total_samples, desc="Progress", unit="sample")

    while num_samples < total_samples:
        if num_samples > prev_num_samples:
            progress_bar.update(num_samples - prev_num_samples)
            prev_num_samples = num_samples
        speakers_of_this_sample = []
        idx_s = np.random.randint(0, np.size(cleanfilenames))
        clean, fs = audioread(cleanfilenames[idx_s])
        base_file_name = os.path.basename(cleanfilenames[idx_s]) # only for libritts
        speaker_name = base_file_name.split('-')[0] # only work for libritts
        speakers_tracker[speaker_name] += 1 # only for libritts
        speakers_of_this_sample.append(speaker_name)

        
        if len(clean)>audio_length:
            clean = clean
        
        else:
            while len(clean)<=audio_length:
                idx_s = idx_s + 1
                if idx_s >= np.size(cleanfilenames)-1:
                    idx_s = np.random.randint(0, np.size(cleanfilenames)) 
                newclean, fs = audioread(cleanfilenames[idx_s])
                base_file_name = os.path.basename(cleanfilenames[idx_s]) # only for libritts
                speaker_name = base_file_name.split('-')[0] # only work for libritts
                speakers_tracker[speaker_name] += 1 # only for libritts
                speakers_of_this_sample.append(speaker_name)
                cleanconcat = np.append(clean, np.zeros(int(fs*silence_length)))
                clean = np.append(cleanconcat, newclean)
        
        noise_of_this_sample = []
        idx_n = np.random.randint(0, np.size(noisefilenames))
        noise, fs = audioread(noisefilenames[idx_n]) #随机读一个噪声
        base_noise_name = os.path.basename(noisefilenames[idx_n]) #
        noise_name = base_noise_name.split('_')[0] # noisyEname
        noise_tracker[noise_name] += 1  #统计噪声的数量 
        noise_of_this_sample.append(noise_name) # add the noise name to the list

        if len(noise)>=len(clean):
            noise = noise[0:len(clean)] #截取噪声的长度
        
        else:
            while len(noise)<=len(clean): #如果噪声的长度小于语音的长度
                idx_n = idx_n + 1 #随机读一个噪声
                if idx_n >= np.size(noisefilenames)-1:
                    idx_n = np.random.randint(0, np.size(noisefilenames))
                newnoise, fs = audioread(noisefilenames[idx_n])
                base_noise_name = os.path.basename(noisefilenames[idx_n]) 
                noise_name = base_noise_name.split('_')[0]
                noise_tracker[noise_name] += 1 #统计噪声的数量
                noiseconcat = np.append(noise, np.zeros(int(fs*silence_length)))#在噪声后面加上一段静音
                noise = np.append(noiseconcat, newnoise)#拼接噪声
                noise_of_this_sample.append(noise_name)# add the noise name to the list
        
        noise = noise[0:len(clean)] #截取噪声的长度
        has_error = False # flag to check if there is an error in writing the files
        for i in range(np.size(SNR)): #对每一个snr进行处理
            clean_snr, noise_snr, noisy_snr = snr_mixer(clean=clean, noise=noise, snr=SNR[i])
            noisyfilename = str(filecounter)+'_SNRdb_'+str(SNR[i]) + '.wav' # only for the amphion task
            cleanfilename = str(filecounter)+'.wav' # only for the amphion task
            noisefilename = str(filecounter)+'_SNRdb_'+str(SNR[i]) + '.wav' # only for the amphion task
            noisypath = os.path.join(noisyspeech_dir, noisyfilename)
            cleanpath = os.path.join(clean_proc_dir, cleanfilename)
            noisepath = os.path.join(noise_proc_dir, noisefilename)
            try:
            	audiowrite(noisy_snr, fs, noisypath, norm=False)
            except Exception as e:
                print("Error: {} encountered with writing to {}. Skipping ...".format(e, noisypath))
                has_error = True
                break
            try: 
            	audiowrite(clean_snr, fs, cleanpath, norm=False)
            except Exception as e:
                print("Error: {} encountered with writing to {}. Skipping ...".format(e, cleanpath))
                os.remove(noisypath)
                has_error = True
                break
            try:
            	audiowrite(noise_snr, fs, noisepath, norm=False)
            except Exception as e:
                print("Error: {} encountered with writing to {}. Skipping ...".format(e, noisepath))
                os.remove(noisypath)
                os.remove(cleanpath)
                has_error = True
                break
            num_samples = num_samples + len(noisy_snr)
        if not has_error:        
            filecounter += 1
        else:
            for speaker_name in speakers_of_this_sample:
                speakers_tracker[speaker_name] -= 1
            for noise_name in noise_of_this_sample:
                noise_tracker[noise_name] -= 1
    with open(speaker_tracker_save_path, 'w') as json_file:
        json.dump(speakers_tracker, json_file, indent=4)            
    with open(noise_tracker_save_path, 'w') as json_file:
        json.dump(noise_tracker, json_file, indent=4)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    # Configurations: read noisyspeech_synthesizer.cfg
    parser.add_argument("--cfg", default = "noisyspeech_synthesizer.cfg", help = "Read noisyspeech_synthesizer.cfg for all the details")
    parser.add_argument("--cfg_str", type=str, default = "noisy_speech" )
    args = parser.parse_args()

    
    cfgpath = os.path.join(os.path.dirname(__file__), args.cfg)
    assert os.path.exists(cfgpath), f"No configuration file as [{cfgpath}]"
    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(cfgpath)
    
    main(cfg._sections[args.cfg_str])
    
