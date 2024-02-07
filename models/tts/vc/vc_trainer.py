# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import json
import json5
import time
import torch
import pyworld as pw
import numpy as np
from tqdm import tqdm
from utils.util import ValueWindow
from torch.utils.data import DataLoader
from models.tts.base.tts_trainer import TTSTrainer
from models.base.base_sampler import VariableSampler

from diffusers import get_scheduler
import torch.nn.functional as F

import accelerate
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from librosa.filters import mel as librosa_mel_fn

from models.tts.vc.ns2_uniamphion import UniAmphionVC
from models.tts.vc.vc_dataset import  VCCollator, VCDataset, batch_by_size
from models.tts.vc.hubert_kmeans import HubertWithKmeans

mel_basis = {}
hann_window = {}
init_mel_and_hann = False

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def diff_loss(pred, target, mask, loss_type="l1"):
    # pred: (B, T, d)
    # target: (B, T, d)
    # mask: (B, T)
    if loss_type == "l1":
        loss = F.l1_loss(pred, target, reduction="none").float() * (
            mask.to(pred.dtype).unsqueeze(-1)
        )
    elif loss_type == "l2":
        loss = F.mse_loss(pred, target, reduction="none").float() * (
            mask.to(pred.dtype).unsqueeze(-1)
        )
    else:
        raise NotImplementedError()
    loss = (torch.mean(loss, dim=-1)).sum() / (mask.to(pred.dtype).sum())
    return loss

def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output

def mel_spectrogram(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):

    global mel_basis, hann_window, init_mel_and_hann
    if not init_mel_and_hann:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[str(fmax) + "_" + str(y.device)] = (
            torch.from_numpy(mel).float().to(y.device)
        )
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
        init_mel_and_hann = True

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    # complex tensor as default, then use view_as_real for future pytorch compatibility
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[str(y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.view_as_real(spec)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec

def interpolate(f0):
    uv = f0 == 0
    if len(f0[~uv]) > 0:
        # interpolate the unvoiced f0
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        uv = uv.astype("float")
        uv = np.min(np.array([uv[:-2], uv[1:-1], uv[2:]]), axis=0)
        uv = np.pad(uv, (1, 1))
    return f0, uv

def extract_world_f0(speech):
    audio = speech.cpu().numpy()
    f0s = []
    for i in range(audio.shape[0]):
        wav = audio[i]
        frame_num = len(wav) // 200
        f0, t = pw.dio(wav.astype(np.float64), 16000, frame_period=12.5)
        f0 = pw.stonemask(wav.astype(np.float64), f0, t, 16000)
        f0, _ = interpolate(f0)
        f0 = torch.from_numpy(f0).to(speech.device)
        f0s.append(f0[:frame_num])
    f0s = torch.stack(f0s, dim=0).float()
    return f0s

class VCTrainer(TTSTrainer):
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg

        cfg.exp_name = args.exp_name
        self._init_accelerator()
        self.accelerator.wait_for_everyone()
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger = get_logger(args.exp_name, log_level="INFO")
        self.time_window = ValueWindow(50)
        if self.accelerator.is_main_process:
            self.logger.info("=" * 56)
            self.logger.info("||\t\t" + "New training process started." + "\t\t||")
            self.logger.info("=" * 56)
            self.logger.info("\n")
            self.logger.debug(f"Using {args.log_level.upper()} logging level.")
            self.logger.info(f"Experiment name: {args.exp_name}")
            self.logger.info(f"Experiment directory: {self.exp_dir}")

        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.accelerator.is_main_process:
            self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # init counts
        self.batch_count: int = 0
        self.step: int = 0
        self.epoch: int = 0
        self.max_epoch = (
            self.cfg.train.max_epoch if self.cfg.train.max_epoch > 0 else float("inf")
        )
        if self.accelerator.is_main_process:
            self.logger.info(
                "Max epoch: {}".format(
                    self.max_epoch if self.max_epoch < float("inf") else "Unlimited"
                )
            )

        # Check values
        if self.accelerator.is_main_process:
            self._check_basic_configs()
            # Set runtime configs
            self.save_checkpoint_stride = self.cfg.train.save_checkpoint_stride
            self.checkpoints_path = [
                [] for _ in range(len(self.save_checkpoint_stride))
            ]
            self.keep_last = [
                i if i > 0 else float("inf") for i in self.cfg.train.keep_last
            ]
            self.run_eval = self.cfg.train.run_eval

        # set random seed
        with self.accelerator.main_process_first():
            start = time.monotonic_ns()
            self._set_random_seed(self.cfg.train.random_seed)
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.debug(
                    f"Setting random seed done in {(end - start) / 1e6:.2f}ms"
                )
                self.logger.debug(f"Random seed: {self.cfg.train.random_seed}")
 
        # setup data_loader
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building dataset...")
            start = time.monotonic_ns()
            self.train_dataloader, self.valid_dataloader = self._build_dataloader()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building dataset done in {(end - start) / 1e6:.2f}ms"
                )

        # setup model
            
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building model...")
            start = time.monotonic_ns()
            self.model, self.w2v = self._build_model()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.debug(self.model)
                self.logger.info(f"Building model done in {(end - start) / 1e6:.2f}ms")
                self.logger.info(
                    f"Model parameters: {self._count_parameters(self.model)/1e6:.2f}M"
                )

        # optimizer & scheduler
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building optimizer and scheduler...")
            start = time.monotonic_ns()
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building optimizer and scheduler done in {(end - start) / 1e6:.2f}ms"
                )

        # accelerate prepare
        if not self.cfg.train.use_dynamic_batchsize:
            with self.accelerator.main_process_first():
                if self.accelerator.is_main_process:
                    self.logger.info("Initializing accelerate...")
            start = time.monotonic_ns()
            (
                self.train_dataloader,
                self.valid_dataloader,
            ) = self.accelerator.prepare(
                self.train_dataloader,
                self.valid_dataloader,
            )

        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key] = self.accelerator.prepare(self.model[key])
        else:
            self.model = self.accelerator.prepare(self.model)

        # prepare w2v to accelerator
        if isinstance(self.w2v, dict):
            for key in self.w2v.keys():
                self.w2v[key] = self.accelerator.prepare(self.w2v[key])
        else:
            self.w2v = self.accelerator.prepare(self.w2v)

        if isinstance(self.optimizer, dict):
            for key in self.optimizer.keys():
                self.optimizer[key] = self.accelerator.prepare(self.optimizer[key])
        else:
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if isinstance(self.scheduler, dict):
            for key in self.scheduler.keys():
                self.scheduler[key] = self.accelerator.prepare(self.scheduler[key])
        else:
            self.scheduler = self.accelerator.prepare(self.scheduler)

        end = time.monotonic_ns()
        if self.accelerator.is_main_process:
            self.logger.info(
                f"Initializing accelerate done in {(end - start) / 1e6:.2f}ms"
            )

        # create criterion
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building criterion...")
            start = time.monotonic_ns()
            self.criterion = self._build_criterion()
            end = time.monotonic_ns()
            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Building criterion done in {(end - start) / 1e6:.2f}ms"
                )

        # TODO: Resume from ckpt need test/debug
        with self.accelerator.main_process_first():
            if args.resume:
                if self.accelerator.is_main_process:
                    self.logger.info("Resuming from checkpoint...")
                start = time.monotonic_ns()
                ckpt_path = self._load_model(
                    self.checkpoint_dir,
                    args.checkpoint_path,
                    resume_type=args.resume_type,
                )
                end = time.monotonic_ns()
                if self.accelerator.is_main_process:
                    self.logger.info(
                        f"Resuming from checkpoint done in {(end - start) / 1e6:.2f}ms"
                    )
                self.checkpoints_path = json.load(
                    open(os.path.join(ckpt_path, "ckpts.json"), "r")
                )

            self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
            if self.accelerator.is_main_process:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
            if self.accelerator.is_main_process:
                self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # save config file path
        self.config_save_path = os.path.join(self.exp_dir, "args.json")

        # Only for TTS tasks
        self.task_type = "TTS"
        if self.accelerator.is_main_process:
            self.logger.info("Task type: {}".format(self.task_type))

    def _init_accelerator(self):
        self.exp_dir = os.path.join(
            os.path.abspath(self.cfg.log_dir), self.args.exp_name
        )
        project_config = ProjectConfiguration(
            project_dir=self.exp_dir,
            logging_dir=os.path.join(self.exp_dir, "log"),
        )
        print("Initializing accelerator......")
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_step,
            log_with=self.cfg.train.tracker,
            project_config=project_config,
        )
        print("Accelerator initialized......")
        if self.accelerator.is_main_process:
            os.makedirs(project_config.project_dir, exist_ok=True)
            os.makedirs(project_config.logging_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        print("Accelerator initing trackers......")
        with self.accelerator.main_process_first():
            self.accelerator.init_trackers(self.args.exp_name)
        print("Accelerator init trackers done......")


    def _build_model(self):
        model = UniAmphionVC(cfg=self.cfg.model)
        w2v = HubertWithKmeans(
            checkpoint_path="/mnt/data3/hehaorui/ckpt/mhubert/mhubert_base_vp_en_es_fr_it3.pt",
            kmeans_path="/mnt/data3/hehaorui/ckpt/mhubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin",
        )
        return model, w2v

    def _build_dataset(self):
        return VCDataset, VCCollator

    def _build_dataloader(self):
        if self.cfg.train.use_dynamic_batchsize:
            print("Use Dynamic Batchsize......")
            Dataset, Collator = self._build_dataset()

            directory_list = [
            '/mnt/data2/wangyuancheng/mls_english/train/audio',
            '/mnt/data2/wangyuancheng/mls_english/dev/audio',
            '/mnt/data4/hehaorui/large_15s',
            '/mnt/data4/hehaorui/medium_15s',
            '/mnt/data4/hehaorui/small_15s',
            ]
            
            train_dataset = Dataset(directory_list)
            train_collate = Collator(self.cfg)
            batch_sampler = batch_by_size(
                train_dataset.num_frame_indices,
                train_dataset.get_num_frames,
                max_tokens=self.cfg.train.max_tokens * self.accelerator.num_processes,
                max_sentences=self.cfg.train.max_sentences
                * self.accelerator.num_processes,
                required_batch_size_multiple=self.accelerator.num_processes,
            )
            np.random.seed(980205)
            np.random.shuffle(batch_sampler)

            batches = [
                x[
                    self.accelerator.local_process_index :: self.accelerator.num_processes
                ]
                for x in batch_sampler
                if len(x) % self.accelerator.num_processes == 0
            ]

            train_loader = DataLoader(
                train_dataset,
                collate_fn=train_collate,
                num_workers=self.cfg.train.dataloader.num_worker,
                batch_sampler=VariableSampler(
                    batches, drop_last=False, use_random_sampler=True
                ),
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )
            self.accelerator.wait_for_everyone()

            valid_dataset = Dataset(["/mnt/data2/wangyuancheng/mls_english/test/audio"])
            valid_collate = Collator(self.cfg)
            batch_sampler = batch_by_size(
                valid_dataset.num_frame_indices,
                valid_dataset.get_num_frames,
                max_tokens=self.cfg.train.max_tokens * self.accelerator.num_processes,
                max_sentences=self.cfg.train.max_sentences
                * self.accelerator.num_processes,
                required_batch_size_multiple=self.accelerator.num_processes,
            )
            batches = [
                x[
                    self.accelerator.local_process_index :: self.accelerator.num_processes
                ]
                for x in batch_sampler
                if len(x) % self.accelerator.num_processes == 0
            ]
            valid_loader = DataLoader(
                valid_dataset,
                collate_fn=valid_collate,
                num_workers=self.cfg.train.dataloader.num_worker,
                batch_sampler=VariableSampler(batches, drop_last=False),
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )
            self.accelerator.wait_for_everyone()

        else:
            print("Use Normal Batchsize......")
            Dataset, Collator = self._build_dataset()
            train_dataset = Dataset(self.cfg, self.cfg.dataset[0], is_valid=False)
            train_collate = Collator(self.cfg)

            train_loader = DataLoader(
                train_dataset,
                shuffle=True,
                collate_fn=train_collate,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.dataloader.num_worker,
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )

            valid_dataset = Dataset(self.cfg, self.cfg.dataset[0], is_valid=True)
            valid_collate = Collator(self.cfg)

            valid_loader = DataLoader(
                valid_dataset,
                shuffle=True,
                collate_fn=valid_collate,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.dataloader.num_worker,
                pin_memory=self.cfg.train.dataloader.pin_memory,
            )
            self.accelerator.wait_for_everyone()

        return train_loader, valid_loader

    def _build_optimizer(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.cfg.train.adam,
        )
        return optimizer

    def _build_scheduler(self):
        lr_scheduler = get_scheduler(
            self.cfg.train.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.train.lr_warmup_steps,
            num_training_steps=self.cfg.train.num_train_steps
        )
        return lr_scheduler

    def _build_criterion(self):
        criterion = torch.nn.L1Loss(reduction="mean")
        return criterion

    def _count_parameters(self, model):
        model_param = 0.0
        if isinstance(model, dict):
            for key, value in model.items():
                model_param += sum(p.numel() for p in model[key].parameters())
        else:
            model_param = sum(p.numel() for p in model.parameters())
        return model_param
    
    def _dump_cfg(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json5.dump(
            self.cfg,
            open(path, "w"),
            indent=4,
            sort_keys=True,
            ensure_ascii=False,
            quote_keys=True,
        )

    def load_model(self, checkpoint):
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

    def _train_step(self, batch):
        train_losses = {}
        train_stats = {}
        device = self.accelerator.device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        speech = batch["speech"]
        ref_speech = batch["ref_speech"]
        with torch.set_grad_enabled(False):
            batch["pitch"] = extract_world_f0(batch["speech"])
            pitch = (batch["pitch"] - batch["pitch"].mean(dim=1, keepdim=True)) / (
                batch["pitch"].std(dim=1, keepdim=True) + 1e-6
            )  
            batch["pitch"] = pitch
            mel = mel_spectrogram(
                speech,
                n_fft=1024,
                num_mels=80,
                sampling_rate=16000,
                hop_size=200,
                win_size=800,
                fmin=0,
                fmax=8000,
            )  # (B, 80, T)
            ref_mel = mel_spectrogram(
                ref_speech,
                n_fft=1024,
                num_mels=80,
                sampling_rate=16000,
                hop_size=200,
                win_size=800,
                fmin=0,
                fmax=8000,
            )  # (B, 80, T')

            mel = mel.transpose(1, 2)
            ref_mel = ref_mel.transpose(1, 2)

            _, content_feature = self.w2v(speech)
            pitch = batch["pitch"]
            mask = batch["mask"]
            ref_mask = batch["ref_mask"]

        with torch.set_grad_enabled(True):
            diff_out = self.model(
            x=mel,
            content_feature=content_feature,
            pitch=pitch,
            x_ref=ref_mel,
            x_mask=mask,
            x_ref_mask=ref_mask,
        )
        total_loss = 0.0

        diff_loss_x0 = diff_loss(diff_out["x0_pred"], mel, mask=mask)
        total_loss += diff_loss_x0
        train_losses["diff_loss_x0"] = diff_loss_x0

        diff_loss_noise = diff_loss(diff_out["noise_pred"], diff_out["noise"], mask=mask)
        total_loss += diff_loss_noise 
        train_losses["diff_loss_noise"] = diff_loss_noise
        train_losses["total_loss"] = total_loss

        self.optimizer.zero_grad()
        self.accelerator.backward(total_loss)
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), 0.5)
        self.optimizer.step()
        self.scheduler.step()

        for item in train_losses:
            train_losses[item] = train_losses[item].item()/pitch.shape[0]

        train_losses["batch_size"] = pitch.shape[0]
        return (train_losses["total_loss"], train_losses, train_stats)

    @torch.inference_mode()
    def _valid_step(self, batch):
        valid_losses = {}
        valid_stats = {}
        device = self.accelerator.device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        with torch.set_grad_enabled(False):
            batch["pitch"] = extract_world_f0(batch["speech"])
            pitch = (batch["pitch"] - batch["pitch"].mean(dim=1, keepdim=True)) / (
                batch["pitch"].std(dim=1, keepdim=True) + 1e-6
            )  
            batch["pitch"] = pitch
            speech = batch["speech"]
            ref_speech = batch["ref_speech"]
            mel = mel_spectrogram(
                speech,
                n_fft=1024,
                num_mels=80,
                sampling_rate=16000,
                hop_size=200,
                win_size=800,
                fmin=0,
                fmax=8000,
            )  # (B, 80, T)
            ref_mel = mel_spectrogram(
                ref_speech,
                n_fft=1024,
                num_mels=80,
                sampling_rate=16000,
                hop_size=200,
                win_size=800,
                fmin=0,
                fmax=8000,
            )  # (B, 80, T')
            mel = mel.transpose(1, 2)
            ref_mel = ref_mel.transpose(1, 2)

            _, content_feature = self.w2v(speech)
            pitch = batch["pitch"]
            mask = batch["mask"]
            ref_mask = batch["ref_mask"]
            diff_out = self.model(
                x=mel,
                content_feature=content_feature,
                pitch=pitch,
                x_ref=ref_mel,
                x_mask=mask,
                x_ref_mask=ref_mask,
            )
        total_loss = 0.0

        diff_loss_x0 = diff_loss(diff_out["x0_pred"], mel, mask=mask)
        total_loss += diff_loss_x0
        valid_losses["diff_loss_x0"] = diff_loss_x0

        # diff loss noise
        diff_loss_noise = diff_loss(
            diff_out["noise_pred"], diff_out["noise"], mask=mask
        )
        total_loss += diff_loss_noise 
        valid_losses["diff_loss_noise"] = diff_loss_noise
        valid_losses["total_loss"] = total_loss

        for item in valid_losses:
            valid_losses[item] = valid_losses[item].item()/pitch.shape[0]

        valid_losses["batch_size"] = pitch.shape[0]
        return (valid_losses["total_loss"], valid_losses, valid_stats)

    @torch.inference_mode()
    def _valid_epoch(self):
        r"""Testing epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].eval()
        else:
            self.model.eval()

        epoch_sum_loss = 0.0
        # epoch_losses = dict()

        for batch in tqdm(
            self.valid_dataloader,
            desc=f"Training Epoch {self.epoch}",
            unit="batch",
            colour="GREEN",
            leave=False,
            dynamic_ncols=True,
            smoothing=0.04,
            disable=not self.accelerator.is_main_process,
        ):
            # Put the data to cuda device
            device = self.accelerator.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            total_loss, _, _ = self._valid_step(batch)
            epoch_sum_loss += total_loss
            # for key, value in valid_losses.items():
            #     epoch_losses[key] = value

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, _

    def _train_epoch(self):
        r"""Training epoch. Should return average loss of a batch (sample) over
        one epoch. See ``train_loop`` for usage.
        """
        if isinstance(self.model, dict):
            for key in self.model.keys():
                self.model[key].train()
        else:
            self.model.train()
        if isinstance(self.w2v, dict):
            for key in self.w2v.keys():
                self.w2v[key].eval()
        else:
            self.w2v.eval()

        epoch_sum_loss: float = 0.0 # total loss
        # epoch_losses: dict = {} # loss dict
        epoch_step: int = 0 # step count

        for batch in tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {self.epoch}",
            unit="batch",
            colour="GREEN",
            leave=False,
            dynamic_ncols=True,
            smoothing=0.04,
            disable=not self.accelerator.is_main_process,
        ):  
            # Put the data to cuda device
            device = self.accelerator.device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            self.model = self.model.to(device)
            self.w2v = self.w2v.to(device)

            # Do training step and BP
            with self.accelerator.accumulate(self.model):
                total_loss, train_losses, _ = self._train_step(batch)
            self.batch_count += 1
            self.step += 1
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss += total_loss
                self.current_loss = total_loss
                if isinstance(train_losses, dict):
                    for key, loss in train_losses.items():
                        # epoch_losses[key] = loss
                        self.accelerator.log(
                            {"Epoch/Train {} Loss".format(key): loss},
                            step=self.step,
                        )

                if (self.accelerator.is_main_process and self.batch_count % 5 == 0):
                    self.echo_log(train_losses, mode="Training")
                
                epoch_step += 1

                self.save_checkpoint() # save checkpont
                
        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, _
    
    def train_loop(self):
        r"""Training loop. The public entry of training process."""
        # Wait everyone to prepare before we move on
        self.accelerator.wait_for_everyone()
        # dump config file
        if self.accelerator.is_main_process:
            self._dump_cfg(self.config_save_path)

        # Wait to ensure good to go
        self.accelerator.wait_for_everyone()
        # stop when meet max epoch or self.cfg.train.num_train_steps
        while self.epoch < self.max_epoch and self.step < self.cfg.train.num_train_steps:
            if self.accelerator.is_main_process:
                self.logger.info("\n")
                self.logger.info("-" * 32)
                self.logger.info("Epoch {}: ".format(self.epoch))
                print("Start training......")
            
            train_total_loss, _ = self._train_epoch()
            if self.accelerator.is_main_process:
                print("Start validating......")
            valid_total_loss, _ = self._valid_epoch()

            if self.accelerator.is_main_process:
                self.logger.info("  |- Train/Loss: {:.6f}".format(train_total_loss/len(self.train_dataloader)))
                self.logger.info("  |- Valid/Loss: {:.6f}".format(valid_total_loss/len(self.train_dataloader)))
            self.accelerator.log(
                {
                    "Epoch/Train Loss": train_total_loss/len(self.train_dataloader),
                    "Epoch/Valid Loss": valid_total_loss/len(self.valid_dataloader),
                },
                step=self.epoch,
            )
            self.epoch += 1

        # Finish training and save final checkpoint
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.accelerator.save_state(
                os.path.join(
                    self.checkpoint_dir,
                    "final_epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                        self.epoch, self.step, train_total_loss
                    ),
                )
            )
        self.accelerator.end_training()
        print("Training finished......")

    def save_checkpoint(self):
        self.accelerator.wait_for_everyone()

        # Check if hit save_checkpoint_stride and run_eval
        run_eval = False
        if self.accelerator.is_main_process:
            save_checkpoint = False
            hit_idx = []
            for i, num in enumerate(self.save_checkpoint_stride):
                if self.step % num == 0: #save every save_checkpoint_stride
                    save_checkpoint = True
                    hit_idx.append(i)
                    run_eval |= self.run_eval[i]

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process and save_checkpoint:
            # 构造检查点文件的路径
            checkpoint_filename = "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                self.epoch, self.step, self.current_loss
            )
            path = os.path.join(self.checkpoint_dir, checkpoint_filename)

            # 保存模型和优化器的状态
            print("Saving state to {}...".format(path))
            self.accelerator.save_state(path)
            print("Finished saving state.")

            # 更新检查点路径记录
            json.dump(
                self.checkpoints_path,
                open(os.path.join(path, "ckpts.json"), "w"),
                ensure_ascii=False,
                indent=4,
            )

            # 移除旧的检查点
            to_remove = []
            for idx in hit_idx:
                self.checkpoints_path[idx].append(path)
                while len(self.checkpoints_path[idx]) > self.keep_last[idx]:
                    to_remove.append((idx, self.checkpoints_path[idx].pop(0)))

            # 查找需要删除的检查点
            total = set()
            for i in self.checkpoints_path:
                total |= set(i)
            do_remove = set()
            for idx, path in to_remove[::-1]:
                if path not in total:
                    do_remove.add(path)

            # 删除旧的检查点
            for path in do_remove:
                shutil.rmtree(path, ignore_errors=True)
                print(f"Removed old checkpoint: {path}")

        # 再次确保所有进程同步
        self.accelerator.wait_for_everyone()



