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

from multiprocessing import Pool, Manager
import random
import torchaudio
import random

NUM_WORKERS = 128
SAMPLE_RATE = 16000

def get_metadata(file_path):
    metadata = torchaudio.info(file_path)
    return file_path, metadata.num_frames

def get_speaker(file_path):
    speaker_id = file_path.split(os.sep)[-3]
    if 'mls' in file_path:
        speaker = 'mls_' + speaker_id
    else:
        speaker = 'libri_' + speaker_id
    return file_path, speaker

def process_files(files):
    metadata_cache_path = '/mnt/data2/hehaorui/ckpt/metadata_cache.json'
    if os.path.exists(metadata_cache_path):
        with open(metadata_cache_path, 'r') as f:
            metadata_cache = json.load(f)
    else:
        metadata_cache = {}
    files_to_process = [file for file in files if file not in metadata_cache]
    if len(files_to_process) != 0:
        with Pool(processes=NUM_WORKERS) as pool:
            for file, num_frames in tqdm(pool.imap_unordered(get_metadata, files_to_process), total=len(files_to_process)):
                metadata_cache[file] = num_frames
        with open(metadata_cache_path, 'w') as f:
            print(f"Saving metadata cache to {metadata_cache_path}")
            json.dump(metadata_cache, f)
    else:
        print(f"skipping processing num_frames, loaded {len(metadata_cache)} files")
    return metadata_cache

def process_speakers(files):
    speaker_cache_path = '/mnt/data2/hehaorui/ckpt/file2speaker.json'
    if os.path.exists(speaker_cache_path):
        with open(speaker_cache_path, 'r') as f:
            speaker_cache = json.load(f)
    else:
        speaker_cache = {}
    files_to_process = [file for file in files if file not in speaker_cache]
    if len(files_to_process) != 0:
        with Pool(processes=NUM_WORKERS) as pool:
            for file, speaker in tqdm(pool.imap_unordered(get_speaker, files_to_process), total=len(files_to_process)):
                speaker_cache[file] = speaker
        with open(speaker_cache_path, 'w') as f:
            print(f"Saving speaker cache to {speaker_cache_path}")
            json.dump(speaker_cache, f)
    else:
        print(f"skipping processing speakers, loaded {len(speaker_cache)} files")
    return speaker_cache

class VCDataset(Dataset):
    def __init__(self, directory_list):
        self.directory_list = directory_list
        self.files = []
        for directory in directory_list:
            print(f"Loading {directory}")
            self.files.extend(self.get_flac_files(directory))
            print(f"Loaded {len(self.files)} files")
            meta_data_cache = process_files(self.files)
            speaker_cache = process_speakers(self.files)

        print(f"Loaded {len(self.files)} files")
        # # random select 500 files for testing
        # self.files = random.sample(self.files, 500)
        random.shuffle(self.files)  # Shuffle the files.
        self.filtered_files, self.all_num_frames, self.index2numframes, self.index2speaker = self.filter_files(meta_data_cache, speaker_cache)
        print(f"Loaded {len(self.filtered_files)} files")
        self.speaker2id = self.create_speaker2id()
        self.num_frame_sorted = np.array(sorted(self.all_num_frames))
        self.num_frame_indices = np.array(
            sorted(
                range(len(self.all_num_frames)), key=lambda k: self.all_num_frames[k]
            )
        )

    def get_flac_files(self, directory):
        flac_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                # flac or wav
                if file.endswith(".flac") or file.endswith(".wav"):
                    flac_files.append(os.path.join(root, file))
        return flac_files

    def get_all_flac(self, directory):
        # 获取目录下的所有子目录
        directories = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        
        # 如果没有子目录，就直接在当前目录中查找
        if not directories:
            return self.get_flac_files(directory)
        with Pool(processes=NUM_WORKERS) as pool:
            results = []
            for result in tqdm(pool.imap_unordered(self.get_flac_files, directories), total=len(directories), desc="Processing"):
                results.extend(result)
        print(f"Found {len(results)} waveform files")
        return results
    
    def get_num_frames(self, index):
        return self.index2numframes[index]
    
    def filter_files(self, metadata_cache, speaker_cache):
        # Filter files
        filtered_files = []
        all_num_frames = []
        index2numframes = {}
        index2speaker = {}
        for file in self.files:
            num_frames = metadata_cache[file]
            if SAMPLE_RATE * 3 <= num_frames <= SAMPLE_RATE * 25:
                filtered_files.append(file)
                all_num_frames.append(num_frames)
                index2speaker[len(filtered_files) - 1] = speaker_cache[file]
                index2numframes[len(filtered_files) - 1] = num_frames
        del metadata_cache
        return filtered_files, all_num_frames, index2numframes, index2speaker
    
    def create_speaker2id(self):
        speaker2id = {}
        unique_id = 0  # 开始的唯一 ID
        print(f"Creating speaker2id from {len(self.index2speaker)} utterences")
        for _, speaker in tqdm(self.index2speaker.items()):
            if speaker not in speaker2id:
                speaker2id[speaker] = unique_id
                unique_id += 1  # 为下一个唯一 speaker 增加 ID
        print(f"Created speaker2id with {len(speaker2id)} speakers")
        return speaker2id
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.filtered_files[idx]
        speech, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        speech = torch.tensor(speech, dtype=torch.float32)
        inputs = self._get_reference_vc(speech, hop_length=200)
        speaker = self.index2speaker[idx]
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

        return {"speech": new_speech, "ref_speech": ref_speech, "ref_mask": ref_mask, "mask": mask}


class VCCollator(BaseCollator):
    def __init__(self, cfg):
        BaseCollator.__init__(self, cfg)

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

