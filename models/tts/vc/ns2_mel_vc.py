from functools import partial
import logging
import multiprocessing as mp
import numpy as np
import os

from torchtts.data.core import audio
from torchtts.data.core import features
from torchtts.data.core.dataset_builder import GeneratorBasedBuilder
from torchtts.data.core.dataset_info import DatasetInfo
from torchtts.utils.data_utils import get_bucket_scheme

logger = logging.getLogger(__name__)


class NS2Dataset(GeneratorBasedBuilder):
    def _info(self):
        return DatasetInfo(
            builder=self,
            description="Codec dataset builder",
            features=features.FeaturesDict(
                {
                    "speech": features.Audio(),
                    "phone_id": features.Tensor(shape=(None,), dtype=np.int64),
                    "duration": features.Tensor(shape=(None,), dtype=np.int64),
                }
            ),
        )

    def _target_suffixs(self):
        return ["speech", "phone_id", "duration"]

    def _split_generators(self):
        path = self._config.get("raw_data", None)
        if path is None:
            raise ValueError("You should specify raw_data in dataset builder")
        return {"train": self._raw_data_generator(split="train", path=path)}

    def _raw_data_generator(self, split, path):
        example_index = 0
        num_workers = self._config.get("preprocess_workers", os.cpu_count())
        if num_workers > 1:
            with mp.Pool(num_workers) as pool:
                for root, _, files in os.walk(path):
                    extract_fn = partial(
                        self._extract_feature,
                        wav_dir=root,
                        audio_config=self._config["audio_config"],
                    )
                    for result in pool.imap_unordered(extract_fn, files):
                        if result is not None:
                            yield f"{example_index:010}", result
                            example_index += 1
        else:
            for root, _, files in os.walk(path):
                for wav_file in files:
                    result = self._extract_feature(
                        wav_file=wav_file,
                        wav_dir=root,
                        audio_config=self._config["audio_config"],
                    )
                    if result is not None:
                        yield f"{example_index:010}", result
                        example_index += 1

    def _data_pipeline(self, datapipe, shuffle):
        shuffle = True
        if shuffle:
            datapipe = datapipe.shuffle(buffer_size=200)

        # filter min length
        min_sample_per_sent = self._config["audio_config"].get("sample_rate", 16000) * 3
        datapipe = datapipe.filter(
            self._filter_min_len, fn_kwargs={"min_len": min_sample_per_sent}
        )

        # filter max length
        max_sample_per_sent = (
            self._config["audio_config"].get("sample_rate", 16000) * 25
        )
        datapipe = datapipe.filter(
            self._filter_max_len, fn_kwargs={"max_len": max_sample_per_sent}
        )

        # # filter duration align frame nums
        hop_length = self._config["audio_config"].get("hop_length", 200)

        datapipe = datapipe.map(
            self._get_reference_vc, fn_kwargs={"hop_length": hop_length}
        )

        batch_size = self._config["batch_size"]
        bucket_step = self._config.get("bucket_step", 1.1)
        bucket_scheme = get_bucket_scheme(batch_size, 8, bucket_step)
        datapipe = datapipe.dynamic_batch(
            group_key_fn=self.get_frames,
            bucket_boundaries=bucket_scheme["boundaries"],
            batch_sizes=bucket_scheme["batch_sizes"],
        )

        # Shuffle on batch
        if shuffle:
            datapipe = datapipe.shuffle(buffer_size=32)
        datapipe = datapipe.collate(
            fn_kwargs={
                "padding_axes": {
                    "speech": 0,
                    "phone_id": 0,
                    "duration": 0,
                    "mask": 0,
                    # "phone_id_mask": 0,
                    "ref_speech": 0,
                    "ref_mask": 0,
                },
                "padding_values": {
                    "speech": 0,
                    "phone_id": 7,
                    "duration": 0,
                    "mask": 0,
                    # "phone_id_mask": 0,
                    "ref_speech": 0,
                    "ref_mask": 0,
                },
            }
        )  # padding <PAD> is 7

        return datapipe

    @staticmethod
    def _filter_min_len(data, min_len):
        return bool(len(data["speech"]) > min_len)

    @staticmethod
    def _filter_max_len(x, max_len):
        return bool(len(x["speech"]) < max_len)

    @staticmethod
    def _filter_min_phone_num(data, min_phone_num):
        return bool(len(data["phone_id"]) > min_phone_num)

    @staticmethod
    def _filter_dur_align_phone_num(data):
        return bool(len(data["phone_id"]) == len(data["duration"]))

    @staticmethod
    def _filter_dur_align_frame_num(data, hop_length):
        return bool(abs(sum(data["duration"]) - len(data["speech"]) // hop_length) <= 3)

    @staticmethod
    def _extract_feature(wav_file, wav_dir, audio_config):
        if os.path.splitext(wav_file)[1] != ".wav":
            return None
        res_type = audio_config.get("res_type", "soxr_hq")
        wav_path = os.path.join(wav_dir, wav_file)
        target_wav_data = audio.load_wav(
            wav_path, audio_config["target_sample_rate"], res_type=res_type
        )
        return {"audio": target_wav_data}

    @staticmethod
    def _align_len(data, hop_length):
        frame_num = sum(data["duration"])
        sample_num = len(data["speech"])

        expected_sample_num = frame_num * hop_length
        if expected_sample_num > sample_num:
            data["speech"] = np.pad(
                data["speech"],
                (0, expected_sample_num - sample_num),
                "constant",
                constant_values=(0, data["speech"][-1]),
            )
        else:
            data["speech"] = data["speech"][:expected_sample_num]

        # add mask
        data["duration"] = np.array(data["duration"])
        data["phone_id"] = np.array(data["phone_id"])

        return data

    @staticmethod
    def get_frames(x):
        return len(x["speech"])

    @staticmethod
    def _get_reference(data, hop_length):
        phone_nums = len(data["phone_id"])
        clip_phone_nums = np.random.randint(
            int(phone_nums * 0.1), int(phone_nums * 0.5) + 1
        )
        clip_phone_nums = max(clip_phone_nums, 2)
        if data["duration"][0] == 0 and clip_phone_nums == 1:
            start_idx = 1
        else:
            start_idx = 0
        end_idx = start_idx + clip_phone_nums
        start_frames = sum(data["duration"][:start_idx])
        end_frames = sum(data["duration"][:end_idx])

        ref_speech = data["speech"][start_frames * hop_length : end_frames * hop_length]

        new_speech = np.append(
            data["speech"][: start_frames * hop_length],
            data["speech"][end_frames * hop_length :],
        )
        new_duration = np.append(
            data["duration"][:start_idx], data["duration"][end_idx:]
        )
        new_phone_id = np.append(
            data["phone_id"][:start_idx], data["phone_id"][end_idx:]
        )

        data["speech"] = new_speech
        data["duration"] = new_duration
        data["phone_id"] = new_phone_id

        data["ref_speech"] = ref_speech
        ref_mask = np.ones((len(data["ref_speech"]) // hop_length))
        data["ref_mask"] = ref_mask
        data["mask"] = np.ones((len(data["speech"]) // hop_length))
        data["phone_id_mask"] = np.ones(data["phone_id"].shape)

        return data

    @staticmethod
    def _get_reference_vc(data, hop_length):
        data["speech"] = np.pad(
            data["speech"],
            (
                0,
                1600 - len(data["speech"]) % 1600,
            ),  
            "constant",
            constant_values=(0, data["speech"][-1]),
        )

        frame_nums = len(data["speech"]) // hop_length
        clip_frame_nums = np.random.randint(
            int(frame_nums * 0.25),
            int(frame_nums * 0.45),
        )
        clip_frame_nums = clip_frame_nums + (frame_nums - clip_frame_nums) % 8
        start_frames = 0
        end_frames = start_frames + clip_frame_nums

        ref_speech = data["speech"][start_frames * hop_length : end_frames * hop_length]
        new_speech = np.append(
            data["speech"][: start_frames * hop_length],
            data["speech"][end_frames * hop_length :],
        )

        data["speech"] = new_speech

        data["ref_speech"] = ref_speech
        ref_mask = np.ones((len(data["ref_speech"]) // hop_length))
        data["ref_mask"] = ref_mask
        data["mask"] = np.ones((len(data["speech"]) // hop_length))

        data["phone_id"] = np.array(data["phone_id"])
        data["duration"] = np.array(data["duration"])

        return data
