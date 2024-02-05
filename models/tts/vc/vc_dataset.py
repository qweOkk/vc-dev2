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

from multiprocessing import Pool
import random
import torchaudio
import random

def get_metadata(file):
    metadata = torchaudio.info(file)
    return file, metadata.num_frames

def process_files(files):
    metadata_cache_path = '/mnt/data2/hehaorui/ckpt/metadata_cache.json'
    if os.path.exists(metadata_cache_path):
        with open(metadata_cache_path, 'r') as f:
            metadata_cache = json.load(f)
    else:
        metadata_cache = {}
    
    # Prepare a list of files to process
    files_to_process = [file for file in files if file not in metadata_cache]

    if len(files_to_process) != 0:
        # Use multiprocessing Pool to process files in parallel
        with Pool(processes=16) as pool:
            for file, num_frames in tqdm(pool.imap_unordered(get_metadata, files_to_process), total=len(files_to_process)):
                metadata_cache[file] = num_frames
        
        # Save updated metadata_cache
        with open(metadata_cache_path, 'w') as f:
            json.dump(metadata_cache, f)
        
    # Filter files
    filtered_files = []
    all_num_frames = []
    index2numframes = {}
    for file in files:
        num_frames = metadata_cache[file]
        if 16000 * 3 <= num_frames <= 16000 * 25:
            filtered_files.append(file)
            all_num_frames.append(num_frames)
            index2numframes[len(filtered_files) - 1] = num_frames
    del metadata_cache
    return filtered_files, all_num_frames, index2numframes

class VCDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        print(f"Loading data from {directory}")
        self.files = self.get_all_flac(directory)
        # if 'dev' in directory:
        #     self.files = random.sample(self.files, 1000)  # Randomly select 1000 files.
        # else:
        #     self.files = random.sample(self.files, 5000)
        
        random.shuffle(self.files)  # Shuffle the files.
        print(f"Loaded {len(self.files)} files")
        self.filtered_files, self.all_num_frames, self.index2numframes = process_files(self.files)

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
                if file.endswith(".flac"):
                    flac_files.append(os.path.join(root, file))
        return flac_files

    def get_all_flac(self, directory):
        # 获取目录下的所有子目录
        directories = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        
        # 如果没有子目录，就直接在当前目录中查找
        if not directories:
            return self.get_flac_files(directory)
        
        # 使用多进程在每个子目录中查找flac文件
        with Pool(processes=16) as pool:
            results = pool.map(self.get_flac_files, directories)
        
        # 合并所有进程的结果
        flac_files = [file for sublist in results for file in sublist]
        return flac_files
    
    def get_num_frames(self, index):
        return self.index2numframes[index]
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.filtered_files[idx]
        speech, _ = librosa.load(file_path, sr=16000)
        speech = torch.tensor(speech, dtype=torch.float32)
        return self._get_reference_vc(speech, hop_length=200)
    
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

        # x_ref: (B, T, d_ref)
        # key_padding_mask: (B, T)

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

