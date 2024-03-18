# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import json
import json5
import torch
import numpy as np
from tqdm import tqdm
from utils.util import ValueWindow
from torch.utils.data import DataLoader
from models.tts.base.tts_trainer import TTSTrainer
from models.base.base_sampler import VariableSampler

from diffusers import get_scheduler

import accelerate
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from models.tts.vc.ns2_uniamphion import UniAmphionVC
# from models.tts.vc.vc_dataset import  VCCollator, VCDataset, batch_by_size
from models.tts.vc.vc_new_dataset import VCCollator, VCDataset, batch_by_size # used on ailab sever
from models.tts.vc.hubert_kmeans import HubertWithKmeans
from models.tts.vc.vc_loss import diff_loss, ConstractiveSpeakerLoss
from models.tts.vc.vc_utils import mel_spectrogram, extract_world_f0


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
        # config noise and speaker
        self.use_noise = self.cfg.trans_exp.use_noise
        self.use_speaker = self.cfg.trans_exp.use_speaker
        if self.accelerator.is_main_process:
            self.logger.info("use_noise: {}".format(self.use_noise))
            self.logger.info("use_speaker: {}".format(self.use_speaker))
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
        self.max_epoch = (self.cfg.train.max_epoch if self.cfg.train.max_epoch > 0 else float("inf"))
        if self.accelerator.is_main_process:
            self.logger.info(
                "Max epoch: {}".format(
                    self.max_epoch if self.max_epoch < float("inf") else "Unlimited"
                )
            )

        # Check values
        if self.accelerator.is_main_process:
            self._check_basic_configs()
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
            self._set_random_seed(self.cfg.train.random_seed)
 
        # setup data_loader
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building dataset...")
            self.train_dataloader, self.valid_dataloader = self._build_dataloader()
            self.speaker_num = len(self.train_dataloader.dataset.speaker2id)
            print("speaker_num", self.speaker_num)
            
        # build model
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building model...")
            self.model, self.w2v = self._build_model()

        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Resume training: {}".format(args.resume))
            if args.resume:
                if self.accelerator.is_main_process:
                    self.logger.info("Resuming from checkpoint...")
                ckpt_path = self._load_model(
                    self.checkpoint_dir,
                    args.checkpoint_path,
                    resume_type=args.resume_type,
                )
                self.checkpoints_path = json.load(
                    open(os.path.join(ckpt_path, "ckpts.json"), "r")
                )

            self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
            if self.accelerator.is_main_process:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
            if self.accelerator.is_main_process:
                self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")


        # optimizer & scheduler
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building optimizer and scheduler...")
            self.optimizer = self._build_optimizer()
            self.scheduler = self._build_scheduler()

        # accelerate prepare
        if not self.cfg.train.use_dynamic_batchsize:
            with self.accelerator.main_process_first():
                if self.accelerator.is_main_process:
                    self.logger.info("Initializing accelerate...")
            (self.train_dataloader, self.valid_dataloader) = self.accelerator.prepare(self.train_dataloader,self.valid_dataloader)

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

        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building criterion...")
            self.criterion = self._build_criterion()

        self.config_save_path = os.path.join(self.exp_dir, "args.json")

        self.task_type = "VC"
        self.speaker_loss_weight = 0.25

        self.contrastive_speaker_loss = ConstractiveSpeakerLoss()

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
        model = UniAmphionVC(cfg=self.cfg.model, use_noise = self.use_noise, use_speaker = self.use_speaker, speaker_num = self.speaker_num)
        w2v = HubertWithKmeans(
            checkpoint_path="/mnt/data3/hehaorui/ckpt/mhubert/mhubert_base_vp_en_es_fr_it3.pt",
            kmeans_path="/mnt/data3/hehaorui/ckpt/mhubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin",
        )
        return model, w2v

    def _build_dataset(self):
        return VCDataset, VCCollator

    def _build_dataloader(self):
        if self.cfg.train.use_dynamic_batchsize:
            np.random.seed(980205)
            if self.accelerator.is_main_process:
                self.logger.info("Use Dynamic Batchsize......")
            train_dataset = VCDataset(self.cfg.trans_exp, TRAIN_MODE=True)
            train_collate = VCCollator(self.cfg)
            batch_sampler = batch_by_size(
                train_dataset.num_frame_indices,
                train_dataset.get_num_frames,
                max_tokens=self.cfg.train.max_tokens * self.accelerator.num_processes,
                max_sentences=self.cfg.train.max_sentences
                * self.accelerator.num_processes,
                required_batch_size_multiple=self.accelerator.num_processes,
            )
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

            valid_dataset = VCDataset(self.cfg.trans_exp, TRAIN_MODE=False)
            valid_collate = VCCollator(self.cfg)
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
            self.logger.info("Use Normal Batchsize......")
            self.logger.info("Exiting......")

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

    def get_state_dict(self):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "step": self.step,
            "epoch": self.epoch,
            "batch_size": self.cfg.train.batch_size,
        }
        return state_dict
    
    def load_model(self, checkpoint):
        self.step = checkpoint["step"]
        self.epoch = checkpoint["epoch"]

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

    def _train_step(self, batch):
        total_loss = 0.0
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

            if self.use_noise:
                noisy_ref_mel = mel_spectrogram(
                    batch["noisy_ref_speech"],
                    n_fft=1024,
                    num_mels=80,
                    sampling_rate=16000,
                    hop_size=200,
                    win_size=800,
                    fmin=0,
                    fmax=8000,
                )
                noisy_ref_mel = noisy_ref_mel.transpose(1, 2)
                with torch.set_grad_enabled(True):
                    diff_out, ref_emb, noisy_ref_emb, am_speaker_loss = self.model(
                        x=mel,
                        content_feature=content_feature,
                        pitch=pitch,
                        x_ref=ref_mel,
                        x_mask=mask,
                        x_ref_mask=ref_mask,
                        x_speaker = batch["speaker_id"],
                        noisy_x_ref=noisy_ref_mel
                    )
            else:
                with torch.set_grad_enabled(True):
                    diff_out, ref_emb, _, am_speaker_loss = self.model(
                            x=mel,
                            content_feature=content_feature,
                            pitch=pitch,
                            x_ref=ref_mel,
                            x_mask=mask,
                            x_ref_mask=ref_mask,
                            x_speaker = batch["speaker_id"]
                        )
                    
                    
         # ---------------Loss Computation---------------
        if self.use_speaker and (am_speaker_loss is not None):
            am_speaker_loss = am_speaker_loss * self.speaker_loss_weight
            total_loss += am_speaker_loss
            train_losses["am_speaker_loss"] = am_speaker_loss

        if self.use_noise:
            # ref_emb: (B, 32, 512)
            # noisy_ref_emb: (B, 32, 512)
            # speaker_ids: (B)
            speaker_ids = batch["speaker_id"]
            ref_emb = torch.mean(ref_emb, dim=1) # (B, 512)
            noisy_speaker_ids = speaker_ids
            noisy_ref_emb = torch.mean(noisy_ref_emb, dim=1) # (B, 512)

            #get all_ref_emb (B+B, 512)
            all_ref_emb = torch.cat([ref_emb, noisy_ref_emb], dim=0)
            assert all_ref_emb.shape[0] == speaker_ids.shape[0] * 2
            #get all_speaker_ids (B+B)
            all_speaker_ids = torch.cat([speaker_ids, noisy_speaker_ids], dim=0)
            assert all_speaker_ids.shape[0] == speaker_ids.shape[0] * 2
            #get contrastive_speaker_loss
            cs_loss = self.contrastive_speaker_loss(all_ref_emb, all_speaker_ids)
            cs_loss = cs_loss * self.speaker_loss_weight
            total_loss += cs_loss
            train_losses["contrastive_speaker_loss"] = cs_loss

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
            train_losses[item] = train_losses[item].item()

        learning_rate = self.optimizer.param_groups[0]['lr']
        formatted_lr = f"{learning_rate:.1e}"
        train_losses['learning_rate'] = formatted_lr
        train_losses["batch_size"] = pitch.shape[0]
        return (train_losses["total_loss"], train_losses, train_stats)

    @torch.inference_mode()
    def _valid_step(self, batch):
        total_loss = 0.0
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
            if self.use_noise:
                noisy_ref_mel = mel_spectrogram(
                    batch["noisy_ref_speech"],
                    n_fft=1024,
                    num_mels=80,
                    sampling_rate=16000,
                    hop_size=200,
                    win_size=800,
                    fmin=0,
                    fmax=8000,
                )
                noisy_ref_mel = noisy_ref_mel.transpose(1, 2)
                diff_out, ref_emb, noisy_ref_emb, am_speaker_loss = self.model(
                    x=mel,
                    content_feature=content_feature,
                    pitch=pitch,
                    x_ref=ref_mel,
                    x_mask=mask,
                    x_ref_mask=ref_mask,
                    x_speaker = batch["speaker_id"],
                    noisy_x_ref=noisy_ref_mel
                )
            else:
                diff_out, ref_emb, _, am_speaker_loss = self.model(
                        x=mel,
                        content_feature=content_feature,
                        pitch=pitch,
                        x_ref=ref_mel,
                        x_mask=mask,
                        x_ref_mask=ref_mask,
                        x_speaker = batch["speaker_id"]
                    )
        if self.use_noise:
            # ref_emb: (B, 32, 512)
            # noisy_ref_emb: (B, 32, 512)
            # speaker_ids: (B)
            speaker_ids = batch["speaker_id"]
            ref_emb = torch.mean(ref_emb, dim=1) # (B, 512)
            noisy_speaker_ids = speaker_ids
            noisy_ref_emb = torch.mean(noisy_ref_emb, dim=1) # (B, 512)

            #get all_ref_emb (B+B, 512)
            all_ref_emb = torch.cat([ref_emb, noisy_ref_emb], dim=0)
            
            #get all_speaker_ids (B+B)
            all_speaker_ids = torch.cat([speaker_ids, noisy_speaker_ids], dim=0)
            #get contrastive _speaker_loss
            cs_loss = self.contrastive_speaker_loss(all_ref_emb, all_speaker_ids)
            cs_loss = cs_loss * self.speaker_loss_weight
            total_loss += cs_loss
            valid_losses["contrastive_speaker_loss"] = cs_loss
        
        if self.use_speaker:
            #validation的时候不计算speaker loss
            pass

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
            valid_losses[item] = valid_losses[item].item()

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

        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, None

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
                        self.accelerator.log(
                            {"Epoch/Train {} Loss".format(key): loss},
                            step=self.step,
                        )

                if (self.accelerator.is_main_process and self.batch_count % 5 == 0):
                    self.echo_log(train_losses, mode="Training")
                
                epoch_step += 1

                self.save_checkpoint() # save checkpont
                
        self.accelerator.wait_for_everyone()

        return epoch_sum_loss, None
    
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
            self.accelerator.wait_for_everyone()
            if isinstance(self.scheduler, dict):
                for key in self.scheduler.keys():
                    self.scheduler[key].step()
            else:
                self.scheduler.step()

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
            checkpoint_filename = "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                self.epoch, self.step, self.current_loss
            )
            path = os.path.join(self.checkpoint_dir, checkpoint_filename)

            print("Saving state to {}...".format(path))
            self.accelerator.save_state(path)
            print("Finished saving state.")

            json.dump(
                self.checkpoints_path,
                open(os.path.join(path, "ckpts.json"), "w"),
                ensure_ascii=False,
                indent=4,
            )

            to_remove = []
            for idx in hit_idx:
                self.checkpoints_path[idx].append(path)
                while len(self.checkpoints_path[idx]) > self.keep_last[idx]:
                    to_remove.append((idx, self.checkpoints_path[idx].pop(0)))

            total = set()
            for i in self.checkpoints_path:
                total |= set(i)
            do_remove = set()
            for idx, path in to_remove[::-1]:
                if path not in total:
                    do_remove.add(path)

            for path in do_remove:
                shutil.rmtree(path, ignore_errors=True)
                print(f"Removed old checkpoint: {path}")

        self.accelerator.wait_for_everyone()



