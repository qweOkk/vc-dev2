# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
import textgrid
from models.base.base_dataset import (
    BaseCollator,
)
from text.cmudict import valid_symbols
from utils.f0 import get_f0_features_using_dio
import torchaudio

class JsonHParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = JsonHParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()


class UniAmphionDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset, is_valid=False):
        assert isinstance(dataset, str)
        processed_data_dir = os.path.join(cfg.preprocess.processed_dir, dataset)
        meta_file = cfg.preprocess.valid_file if is_valid else cfg.preprocess.train_file
        self.metafile_path = os.path.join(processed_data_dir, meta_file)
        self.metadata = self.get_metadata()
        self.mfa_dictionary_path = os.path.join(processed_data_dir, "mfa_dictionary.dict")

        self.cfg = cfg

        if cfg.preprocess.use_spkid:
            self.utt2spkid = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)
                self.utt2spkid[utt] = utt_info["speaker"]

        if cfg.preprocess.use_duration:
            self.utt2duration_path = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)

                self.utt2duration_path[utt] = os.path.join(
                    cfg.preprocess.processed_dir,
                    dataset,
                    cfg.preprocess.duration_dir,  # duration
                    utt_info["speaker"],
                    uid + ".TextGrid",
                )

        # for cross reference
        if cfg.preprocess.use_cross_reference:
            self.spkid2utt = {}
            for utt_info in self.metadata:
                dataset = utt_info["Dataset"]
                uid = utt_info["Uid"]
                utt = "{}_{}".format(dataset, uid)
                spkid = utt_info["speaker"]
                if spkid not in self.spkid2utt:
                    self.spkid2utt[spkid] = []
                self.spkid2utt[spkid].append(utt)

        # get phone to id / id to phone map
        self.phone2id, self.id2phone = self.get_phone_map()

        self.all_num_frames = []
        for i in range(len(self.metadata)):
            self.all_num_frames.append(self.metadata[i]["num_frames"])
        self.num_frame_sorted = np.array(sorted(self.all_num_frames))
        self.num_frame_indices = np.array(
            sorted(
                range(len(self.all_num_frames)), key=lambda k: self.all_num_frames[k]
            )
        )

    def __len__(self):
        return len(self.metadata)

    def get_dataset_name(self):
        return self.metadata[0]["Dataset"]

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print("metadata len: ", len(metadata))

        return metadata

    # 1
    def parse_phones_and_create_map(self, filename):
        phones_set = set()
        phone_to_id = {}
        current_id = 0

        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                # 提取 phone
                phones = line.split('\t')[0].split('|')
                
                for phone in phones:
                    # 添加到集合和映射
                    if phone not in phones_set:
                        phones_set.add(phone)
                        phone_to_id[phone] = current_id
                        current_id += 1

        return phones_set, phone_to_id
    
    # 2
    def get_duration_phone_id_start_end(self, textgrid_path, phone_to_id):
        tg = textgrid.TextGrid.fromFile(textgrid_path)

        # 假设您的 phones 在第一个层 (Tier)
        phone_tier = tg[0]

        durations = []
        phone_ids = []
        starts = []
        ends = []

        for interval in phone_tier:
            start = interval.minTime
            end = interval.maxTime
            phone = interval.mark
            duration = end - start
            phone_id = phone_to_id.get(phone, -1)  # 如果找不到 phone，返回 -1

            durations.append(duration)
            phone_ids.append(phone_id)
            starts.append(start)
            ends.append(end)

        return durations, phone_ids, starts, ends


    def trim_silence_from_audio(speech, duration, starts, ends, phone_ids, phone_to_id, sr = 160000):
        # 找到第一个和最后一个非 silence 的 phone
        non_silence_indices = [i for i, p_id in enumerate(phone_ids) if (p_id != -1 and p_id!=phone_to_id[""])]
        if not non_silence_indices:
            return speech  

        start_trim = starts[non_silence_indices[0]]
        end_trim = ends[non_silence_indices[-1]]

        # 将时间转换为样本索引
        start_sample = int(start_trim * sr)
        end_sample = int(end_trim * sr)

        # 裁剪音频
        trimmed_speech = speech[:, start_sample:end_sample]
        trimmed_duration = duration[start_trim:end_trim]
        trimmed_phone_ids = phone_ids[start_trim:end_trim]
        return trimmed_speech, trimmed_duration, trimmed_phone_ids
    
    def align_length(self, speech, pitch, duration):
        # 获取 trimmed_speech 的总长度（毫秒）
        trimmed_speech_length_ms = (trimmed_speech.shape[1] / self.sr) * 1000

        # 获取 duration 的总和（毫秒）
        duration_sum = sum(duration)
        min_len = min(trimmed_speech_length_ms, duration_sum)
        # 对齐 trimmed_speech 的长度到 duration 的总和
        if trimmed_speech_length_ms > min_len:
            end_sample = int((duration_sum / 1000) * self.sr)
            trimmed_speech = trimmed_speech[:, :end_sample]
        elif duration_sum > min_len:
            excess_duration = duration_sum - trimmed_speech_length_ms
            duration[-1] -= excess_duration
            assert duration[-1] >= 0

        pitch_len = len(pitch)
        if pitch_len >= min_len:
            pitch = pitch[:min_len]
        else:
            pitch = np.pad(pitch, (0, min_len - pitch_len), mode="edge")

        return speech, pitch, duration
    

    def get_target_and_reference(self, speech, pitch, duration, phone_id, frame_nums):
        phone_nums = len(phone_id)
        clip_phone_nums = np.random.randint(int(phone_nums * 0.1), int(phone_nums * 0.5) + 1)
        clip_phone_nums = max(clip_phone_nums, 1)
        assert clip_phone_nums < phone_nums and clip_phone_nums >= 1

        if self.cfg.preprocess.clip_mode == "mid":
            start_idx = np.random.randint(0, phone_nums - clip_phone_nums)
        elif self.cfg.preprocess.clip_mode == "start":
            if duration[0] == 0 and clip_phone_nums == 1:
                start_idx = 1
            else:
                start_idx = 0
        else:
            assert self.cfg.preprocess.clip_mode in ["mid", "start"]

        end_idx = start_idx + clip_phone_nums
        start_frames = sum(duration[:start_idx])
        end_frames = sum(duration[:end_idx])

        # 调整 speech 张量
        new_speech = np.concatenate((speech[:, :start_frames], speech[:, end_frames:]), axis=1)
        ref_speech = speech[:, start_frames:end_frames]

        # 调整 pitch, duration, phone_id
        new_pitch = np.append(pitch[:start_frames], pitch[end_frames:])
        ref_pitch = pitch[start_frames:end_frames]

        new_duration = np.append(duration[:start_idx], duration[end_idx:])
        ref_duration = duration[start_idx:end_idx]

        new_phone_id = np.append(phone_id[:start_idx], phone_id[end_idx:])
        ref_phone_id = phone_id[start_idx:end_idx]

        # 调整 frame_nums
        new_frame_nums = frame_nums - (end_frames - start_frames)
        ref_frame_nums = end_frames - start_frames

        return {
            "speech": new_speech,
            "ref_speech": ref_speech,
            "pitch": new_pitch,
            "ref_pitch": ref_pitch,
            "duration": new_duration,
            "ref_duration": ref_duration,
            "phone_id": new_phone_id,
            "ref_phone_id": ref_phone_id,
            "frame_nums": new_frame_nums,
            "ref_frame_nums": ref_frame_nums,
        }


    def __getitem__(self, index):
        single_feature = dict()
        utt_info = self.metadata[index]
        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        # get speaker_id
        spkid = self.utt2spkid[utt]

        # get speech(.flac files)
        speech, sr = torchaudio.load(utt_info["Path"])[0]
        # resample to 16k
        if sr != 16000:
            speech = torchaudio.transforms.Resample(sr, 16000)(speech)

        # get phone to id map from dictionary.dict
        phones_set, phone_to_id = self.parse_phones_and_create_map(self.mfa_dictionary_path)
        
        # get duration/phone_id from textgrid
        textgrid_path = self.utt2duration_path[utt]
        durations, phone_ids, starts, ends= self.get_duration_phone_id_start_end(textgrid_path, phone_to_id)

        # trim silence at begginning and end in speech based on durations and phone_ids
        trimmed_speech, trimmed_duration, trimmed_phone_ids = self.trim_silence_from_audio(speech, durations, starts, ends, phone_ids, phone_to_id, sr = 160000)
            
        # get frame_nums
        frame_nums = trimmed_speech.shape[1]

        # get pitch
        trimmed_pitch = get_f0_features_using_dio(trimmed_speech, self.cfg)

        # align length
        speech, pitch, duration = self.align_length(trimmed_speech, trimmed_pitch, trimmed_duration)
        frame_nums = speech.shape[1]

        # get target and reference
        out = self.get_target_and_reference(speech, pitch, duration, phone_id, frame_nums)
        speech, ref_speech = out["speech"], out["ref_speech"]
        pitch, ref_pitch = out["pitch"], out["ref_pitch"]
        duration, ref_duration = out["duration"], out["ref_duration"]
        phone_id, ref_phone_id = out["phone_id"], out["ref_phone_id"]
        frame_nums, ref_frame_nums = out["frame_nums"], out["ref_frame_nums"]

        assert len(phone_id) == len(duration)
        phone_id_frame = []
        for i in range(len(phone_id)):
            phone_id_frame.extend([phone_id[i] for _ in range(duration[i])])
        phone_id_frame = np.array(phone_id_frame)

        assert len(ref_phone_id) == len(ref_duration)
        ref_phone_id_frame = []
        for i in range(len(ref_phone_id)):
            ref_phone_id_frame.extend([ref_phone_id[i] for _ in range(ref_duration[i])])
        ref_phone_id_frame = np.array(ref_phone_id_frame)

        single_feature.update(
            {
                "speech": speech,
                "frame_nums": frame_nums,
                "pitch": pitch,
                "duration": duration,
                "phone_id": phone_id,
                "phone_id_frame": phone_id_frame,
                "ref_speech": ref_speech,
                "ref_frame_nums": ref_frame_nums,
                "ref_pitch": ref_pitch,
                "ref_duration": ref_duration,
                "ref_phone_id": ref_phone_id,
                "ref_phone_id_frame": ref_phone_id_frame,
                "spkid": spkid,
            }
        )

        return single_feature






class NS2Collator(BaseCollator):
    def __init__(self, cfg):
        BaseCollator.__init__(self, cfg)

    def __call__(self, batch):
        packed_batch_features = dict()

        # code: (B, 16, T)
        # frame_nums: (B,)   not used
        # pitch: (B, T)
        # duration: (B, N)
        # phone_id: (B, N)
        # phone_id_frame: (B, T)
        # ref_code: (B, 16, T')
        # ref_frame_nums: (B,)   not used
        # ref_pitch: (B, T)   not used
        # ref_duration: (B, N')   not used
        # ref_phone_id: (B, N')   not used
        # ref_phone_frame: (B, T')   not used
        # spkid: (B,)   not used
        # phone_mask: (B, N)
        # mask: (B, T)
        # ref_mask: (B, T')

        for key in batch[0].keys():
            if key == "phone_id":
                phone_ids = [torch.LongTensor(b["phone_id"]) for b in batch]
                phone_masks = [torch.ones(len(b["phone_id"])) for b in batch]
                packed_batch_features["phone_id"] = pad_sequence(
                    phone_ids,
                    batch_first=True,
                    padding_value=0,
                )
                packed_batch_features["phone_mask"] = pad_sequence(
                    phone_masks,
                    batch_first=True,
                    padding_value=0,
                )
            elif key == "phone_id_frame":
                phone_id_frames = [torch.LongTensor(b["phone_id_frame"]) for b in batch]
                masks = [torch.ones(len(b["phone_id_frame"])) for b in batch]
                packed_batch_features["phone_id_frame"] = pad_sequence(
                    phone_id_frames,
                    batch_first=True,
                    padding_value=0,
                )
                packed_batch_features["mask"] = pad_sequence(
                    masks,
                    batch_first=True,
                    padding_value=0,
                )
            elif key == "ref_code":
                ref_codes = [
                    torch.from_numpy(b["ref_code"]).transpose(0, 1) for b in batch
                ]
                ref_masks = [torch.ones(max(b["ref_code"].shape[1], 1)) for b in batch]
                packed_batch_features["ref_code"] = pad_sequence(
                    ref_codes,
                    batch_first=True,
                    padding_value=0,
                ).transpose(1, 2)
                packed_batch_features["ref_mask"] = pad_sequence(
                    ref_masks,
                    batch_first=True,
                    padding_value=0,
                )
            elif key == "code":
                codes = [torch.from_numpy(b["code"]).transpose(0, 1) for b in batch]
                masks = [torch.ones(max(b["code"].shape[1], 1)) for b in batch]
                packed_batch_features["code"] = pad_sequence(
                    codes,
                    batch_first=True,
                    padding_value=0,
                ).transpose(1, 2)
                packed_batch_features["mask"] = pad_sequence(
                    masks,
                    batch_first=True,
                    padding_value=0,
                )
            elif key == "pitch":
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=50.0
                )
            elif key == "duration":
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )
            elif key == "frame_nums":
                packed_batch_features["frame_nums"] = torch.LongTensor(
                    [b["frame_nums"] for b in batch]
                )
            elif key == "ref_frame_nums":
                packed_batch_features["ref_frame_nums"] = torch.LongTensor(
                    [b["ref_frame_nums"] for b in batch]
                )
            else:
                pass

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
