import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from models.base.base_dataset import (
    BaseCollator,
)

from multiprocessing import Pool, Lock
import random
import torchaudio
import rir_generator as rir
import pandas as pd


NUM_WORKERS = 64
lock = Lock()  # 创建一个全局锁
SAMPLE_RATE = 16000

def get_duration(file_path):
    duration = librosa.get_duration(path=file_path, sr=SAMPLE_RATE)
    return file_path, duration

# def get_duration(file_path):
#     duration = torchaudio.info(file_path).num_frames / SAMPLE_RATE
#     return file_path, duration

def get_speaker(file_path):
    speaker_id = file_path.split(os.sep)[-3]
    if 'mls' in file_path:
        speaker = 'mls_' + speaker_id
    else:
        speaker = 'libri_' + speaker_id
    return file_path, speaker

def safe_write_to_file(data, file_path, mode='w'):
    try:
        with lock, open(file_path, mode, encoding='utf-8') as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
    except IOError as e:
        print(f"Error writing to {file_path}: {e}")


class VCDataset(Dataset):
    def __init__(self, args, TRAIN_MODE=True):
        print(f"Initializing VCDataset")

        if TRAIN_MODE:
            dataset_list = args.dataset_list
            dataset_cache_dir = args.cache_dir
        else:
            dataset_list = args.test_dataset_list
            dataset_cache_dir = args.cache_dir

        random.shuffle(dataset_list)

        os.makedirs(dataset_cache_dir, exist_ok=True)
        # create dataset2dir

        self.dataset2dir = {
            'mls_english_opus': '/mnt/data2/hehaorui/mls_english_opus/train/audio',
            'librilight_small': '/mnt/data4/hehaorui/small_15s',
            'librilight_medium': '/mnt/data4/hehaorui/medium_15s',
            'librilight_large': '/mnt/data4/hehaorui/large_15s',
            'mls_test':'/mnt/data2/wangyuancheng/mls_english/test/audio',
        }

        self.use_speaker = args.use_speaker
        self.use_noise = args.use_noise
        print(f"Using speaker: {self.use_speaker}, using noise: {self.use_noise}")
        print(f"Using {NUM_WORKERS} workers")

        self.dataset_list = dataset_list
        self.meta_data_cache = []
        self.meta_data_cache_path = os.path.join(dataset_cache_dir, "MAIN_metadata_cache.csv")

        print(f"Loading {len(dataset_list)} datasets: {dataset_list}")
        for dataset in dataset_list:
            if dataset not in self.dataset2dir:
                raise ValueError(f"Unknown dataset: {dataset}")

            dataset_cache_path = os.path.join(dataset_cache_dir, f"{dataset}_metadata_cache.csv")

            if os.path.exists(dataset_cache_path):
                print(f"Loading metadata_cache from {dataset_cache_path}")
                dataset_meta_data_cache = pd.read_csv(dataset_cache_path, encoding='utf-8')
                print(f"Loaded {len(dataset_meta_data_cache)} metadata_cache")
            else:
                print(f"Creating metadata_cache for {dataset}")
                dataset_meta_data_cache = self.create_metadata_cache(dataset, dataset_cache_dir)
                print(f"Saved metadata cache to {dataset_cache_path}")
            self.meta_data_cache.append(dataset_meta_data_cache)
        
        self.meta_data_cache = pd.concat(self.meta_data_cache, ignore_index=True) #合并所有的metadata_cache
        
        print(f"Loaded {len(self.meta_data_cache)} metadata_cache")
        self.meta_data_cache = self.meta_data_cache.sample(frac=1.0).reset_index(drop=True) #打乱顺序
        self.meta_data_cache.to_csv(self.meta_data_cache_path, index=False, encoding='utf-8') #保存到文件
        print(f"Saved metadata cache to {self.meta_data_cache_path}")

        # create speaker2speaker_id
        self.speaker2id = self.create_speaker2id()
        self.all_num_frames = self.meta_data_cache['duration'].apply(lambda x: int(x * SAMPLE_RATE)).to_list()
        self.num_frame_sorted = np.array(sorted(self.all_num_frames))
        self.num_frame_indices = np.array(sorted(range(len(self.all_num_frames)), key=lambda k: self.all_num_frames[k]))
        if self.use_noise:
            if TRAIN_MODE:
                self.noise_filenames = self.get_all_audios(args.noise_dir) #rel_paths
                # rel_paths to all paths
                self.noise_filenames = [os.path.join(args.noise_dir, rel_path) for rel_path in self.noise_filenames]
            else:
                self.noise_filenames = self.get_all_audios(args.test_noise_dir)
                self.noise_filenames = [os.path.join(args.test_noise_dir, rel_path) for rel_path in self.noise_filenames]

    def create_metadata_cache(self, dataset, cache_dir):
        dataset_relpath2duration_path = os.path.join(cache_dir, f"{dataset}_relpath2duration.json")
        dataset_relpath2speaker_path = os.path.join(cache_dir, f"{dataset}_relpath2speaker.json")
        dataset_index2relpath_path = os.path.join(cache_dir, f"{dataset}_index2relpath.json")
        dataset_meta_data_cache_path = os.path.join(cache_dir, f"{dataset}_metadata_cache.csv")

        if os.path.exists(dataset_relpath2duration_path) and os.path.exists(
                dataset_relpath2speaker_path) and os.path.exists(dataset_index2relpath_path):
            print(f"Loading cache for {dataset}")
            with open(dataset_relpath2duration_path, 'r', encoding='utf-8') as f:
                relpath2duration = json.load(f)
            with open(dataset_relpath2speaker_path, 'r', encoding='utf-8') as f:
                relpath2speaker = json.load(f)
            with open(dataset_index2relpath_path, 'r', encoding='utf-8') as f:
                index2relpath = json.load(f)
            print(f"Loaded cache for {dataset} with {len(relpath2duration)} files")
        else:
            print(f"Creating cache for {dataset}")
            relpath2duration = {}
            relpath2speaker = {}
            index2relpath = {}

            audio_rel_paths = self.get_audio_files(self.dataset2dir[dataset])
            random.shuffle(audio_rel_paths)
            print(f"Loaded {len(audio_rel_paths)} files from {dataset}")

            print(f"Generating cache for {dataset}")
            relpath2duration, relpath2speaker, index2relpath = self.get_duration_speaker_and_filter(
                dataset, audio_rel_paths)
            print(f"Generated cache for {dataset} with {len(relpath2duration)} files")
            print(f"Saving cache for {dataset}")
            self.save_cache_files(dataset_relpath2duration_path, dataset_relpath2speaker_path,
                                  dataset_index2relpath_path, relpath2duration, relpath2speaker, index2relpath)
            print(f"Saved cache for {dataset}")

        meta_datas = []
        print(f"Generating metadata cache for {dataset}")
        for idx, relpath in tqdm(index2relpath.items()):
            temp_item = {
                'uid': f"{dataset}#{str(idx)}",
                'relpath': relpath,
                'duration': relpath2duration[relpath],
                'speaker': relpath2speaker[relpath]
            }
            meta_datas.append(temp_item)
        dataset_meta_data_cache = pd.DataFrame(meta_datas)
        dataset_meta_data_cache.to_csv(dataset_meta_data_cache_path, index=False, encoding='utf-8')
        return dataset_meta_data_cache

    def save_cache_files(self, relpath2duration_path, relpath2speaker_path, index2relpath_path,
                         relpath2duration, relpath2speaker, index2relpath):
        safe_write_to_file(relpath2duration, relpath2duration_path)
        print(f"Saved relpath2duration to {relpath2duration_path}")
        safe_write_to_file(relpath2speaker, relpath2speaker_path)
        print(f"Saved relpath2speaker to {relpath2speaker_path}")
        safe_write_to_file(index2relpath, index2relpath_path)
        print(f"Saved index2relpath to {index2relpath_path}")

    def get_duration_speaker_and_filter(self, dataset, audio_rel_paths):
        print(f"Processing metadata...")
        rel_path2duration = {}
        rel_path2speaker = {}
        idx2rel_path = {}
        # relative_path to full_path
        base_dir = self.dataset2dir[dataset]
        full_paths = [os.path.join(base_dir, rel_path) for rel_path in audio_rel_paths]
        # # sample 1000 files to get duration
        # print(f"Sampling 1000 files to get duration")
        # full_paths = random.sample(full_paths, 1000)
        with Pool(processes=NUM_WORKERS) as pool:
            results = list(tqdm(pool.imap_unordered(get_duration, full_paths), total=len(audio_rel_paths)))
        
        idx = 0
        print(f"Filtering files with duration between 3.0 and 25.0 seconds")
        for file, duration in tqdm(results):
            if duration > 3.0 and duration < 25.0:
                rel_path = os.path.relpath(file, base_dir)
                rel_path2duration[rel_path] = duration
                speaker_id = file.split(os.sep)[-3]
                #dataset+speaker_id
                speaker = f"{dataset}_{speaker_id}"
                rel_path2speaker[rel_path] = speaker
                idx2rel_path[idx] = rel_path
                idx += 1
        return rel_path2duration, rel_path2speaker, idx2rel_path

    def get_audio_files(self, directory):
        audio_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.flac', '.wav', '.opus')):
                    rel_path = os.path.relpath(os.path.join(root, file), directory)
                    audio_files.append(rel_path)
        return audio_files

    def get_all_audios(self, directory):
        directories = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        if not directories:
            return self.get_audio_files(directory)
        with Pool(processes=NUM_WORKERS) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(self.get_audio_files, directories), total=len(directories), desc="Processing"):
                results.extend(result)
        print(f"Found {len(results)} waveform files")
        return results
    
    def get_num_frames(self, index):
        # get_num_frames(durations) by index
        duration = self.meta_data_cache['duration'][index]
        num_frames = int(duration * SAMPLE_RATE)
        return num_frames
    
    def create_speaker2id(self):
        all_speakers = self.meta_data_cache['speaker'].unique()
        speaker2id = {}
        for idx, speaker in enumerate(all_speakers):
            speaker2id[speaker] = idx
        return speaker2id
    
    def snr_mixer(self, clean, noise, snr):
        # Normalizing to -25 dB FS
        rmsclean = (clean**2).mean()**0.5
        epsilon = 1e-10
        rmsclean = max(rmsclean, epsilon)
        scalarclean = 10 ** (-25 / 20) / rmsclean
        clean = clean * scalarclean

        rmsnoise = (noise**2).mean()**0.5
        scalarnoise = 10 ** (-25 / 20) /rmsnoise
        noise = noise * scalarnoise
        rmsnoise = (noise**2).mean()**0.5
        
        # Set the noise level for a given SNR
        noisescalar = np.sqrt(rmsclean / (10**(snr/20)) / rmsnoise)
        noisenewlevel = noise * noisescalar
        noisyspeech = clean + noisenewlevel
        noisyspeech_tensor = torch.tensor(noisyspeech, dtype=torch.float32)
        return noisyspeech_tensor
    
    def add_noise(self, clean):
        # self.noise_filenames: list of noise files
        random_idx = np.random.randint(0, np.size(self.noise_filenames))
        selected_noise_file = self.noise_filenames[random_idx]
        noise, _ = librosa.load(selected_noise_file, sr=SAMPLE_RATE)
        clean = clean.cpu().numpy()
        if len(noise)>=len(clean):
            noise = noise[0:len(clean)] #截取噪声的长度
        else:
            while len(noise)<=len(clean): #如果噪声的长度小于语音的长度
                random_idx = (random_idx + 1)%len(self.noise_filenames) #随机读一个噪声
                newnoise, fs = librosa.load(selected_noise_file, sr=SAMPLE_RATE)
                noiseconcat = np.append(noise, np.zeros(int(fs * 0.2)))#在噪声后面加上0.2静音
                noise = np.append(noiseconcat, newnoise)#拼接噪声
        noise = noise[0:len(clean)] #截取噪声的长度
        #随机sample一个小于20大于0的随机数
        snr = random.uniform(0.0,20.0)
        noisyspeech = self.snr_mixer(clean=clean, noise=noise, snr=snr) #根据随机的SNR级别，混合生成带噪音频
        del noise
        return noisyspeech
    
    def add_reverb(self, speech):
        room_dim = [np.random.uniform(1, 12) for _ in range(3)]  # [length, width, height]
        mic_pos = [np.random.uniform(0, dim) for dim in room_dim] # 随机选择麦克风位置
        distance = np.random.normal(2, 4) # 确定声源与麦克风的距离
        while distance <= 0 or distance > 5:
            distance = np.random.normal(2, 4)
        source_pos = [mic_pos[0] + distance, mic_pos[1], mic_pos[2]] # 随机选择声源位置，确保它在以麦克风为中心的球内
        rt60 = np.random.uniform(0.05, 1.0) # 随机选择RT60值
        try: 
            rir_filter = rir.generate(
                c=340,                  # 声速
                fs=SAMPLE_RATE,
                r=[mic_pos],            # 麦克风位置
                s=source_pos,           # 声源位置
                L=room_dim,             # 房间尺寸
                reverberation_time=rt60,# RT60值
                nsample=4096,           # IR长度
            )
            # 应用混响
            speech_reverb = np.convolve(speech.cpu().numpy(), rir_filter[:, 0], mode='same')
            speech = torch.tensor(speech_reverb, dtype=torch.float32)
            return speech
        except:
            return speech #如果遇到ValueError: s is outside the room，直接返回没加混响的声音

    def __len__(self):
        return len(self.meta_data_cache)

    def __getitem__(self, idx):
        # Get the file rel path
        file_rel_path = self.meta_data_cache['relpath'][idx]
        # Get the dataset from cache uid
        dataset = self.meta_data_cache['uid'][idx].split('#')[0]
        # Get the full file path
        file_path = os.path.join(self.dataset2dir[dataset], file_rel_path)
        # Load the speech
        speech, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        speech = torch.tensor(speech, dtype=torch.float32)
        inputs = self._get_reference_vc(speech, hop_length=200)
        # Get the speaker id
        speaker = self.meta_data_cache['speaker'][idx]
        speaker_id = self.speaker2id[speaker]
        inputs["speaker_id"] = speaker_id
        return inputs
    
    def _get_reference_vc(self, speech, hop_length):
        pad_size = 1600 - speech.shape[0] % 1600
        speech = torch.nn.functional.pad(speech, (0, pad_size))

        frame_nums = speech.shape[0] // hop_length
        clip_frame_nums = np.random.randint(int(frame_nums * 0.25), int(frame_nums * 0.45))
        clip_frame_nums += (frame_nums - clip_frame_nums) % 8
        start_frames, end_frames = 0, clip_frame_nums

        ref_speech = speech[start_frames * hop_length : end_frames * hop_length]
        new_speech = torch.cat((speech[:start_frames * hop_length], speech[end_frames * hop_length:]), 0)

        ref_mask = torch.ones(len(ref_speech) // hop_length)
        mask = torch.ones(len(new_speech) // hop_length)

        if not self.use_noise:
            return {"speech": new_speech, "ref_speech": ref_speech, "ref_mask": ref_mask, "mask": mask}
        else:
            noisy_ref_speech = self.add_noise(ref_speech) # 添加噪声
            noisy_ref_speech_with_reverb = self.add_reverb(noisy_ref_speech) #进混响
            return {"speech": new_speech, "ref_speech": ref_speech, "noisy_ref_speech": noisy_ref_speech_with_reverb, "ref_mask": ref_mask, "mask": mask}

class VCCollator(BaseCollator):
    def __init__(self, cfg):
        BaseCollator.__init__(self, cfg)
        self.use_noise = cfg.trans_exp.use_noise

    def __call__(self, batch):
        packed_batch_features = dict()

        # Function to handle tensor copying
        def process_tensor(data, dtype=torch.float32):
            if isinstance(data, torch.Tensor):
                return data.clone().detach()
            else:
                return torch.tensor(data, dtype=dtype)

        # Process 'speech' data
        speeches = [process_tensor(b['speech']) for b in batch]
        packed_batch_features['speech'] = pad_sequence(speeches, batch_first=True, padding_value=0)

        # Process 'ref_speech' data
        ref_speeches = [process_tensor(b['ref_speech']) for b in batch]
        packed_batch_features['ref_speech'] = pad_sequence(ref_speeches, batch_first=True, padding_value=0)

        # Process 'mask' data
        masks = [process_tensor(b['mask']) for b in batch]
        packed_batch_features['mask'] = pad_sequence(masks, batch_first=True, padding_value=0)

        # Process 'ref_mask' data
        ref_masks = [process_tensor(b['ref_mask']) for b in batch]
        packed_batch_features['ref_mask'] = pad_sequence(ref_masks, batch_first=True, padding_value=0)

        # Process 'speaker_id' data
        speaker_ids = [process_tensor(b['speaker_id'], dtype=torch.int64) for b in batch]
        packed_batch_features['speaker_id'] = torch.stack(speaker_ids, dim=0)

        if self.use_noise:
            # Process 'noisy_ref_speech' data
            noisy_ref_speeches = [process_tensor(b['noisy_ref_speech']) for b in batch]
            packed_batch_features['noisy_ref_speech'] = pad_sequence(noisy_ref_speeches, batch_first=True, padding_value=0)
        return packed_batch_features


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
    indices,
    num_tokens_fn,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    bsz_mult = required_batch_size_multiple

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert (
            sample_len <= max_tokens
        ), "sentence at index {} of size {} exceeds max_tokens " "limit of {}!".format(
            idx, sample_len, max_tokens
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches

