# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import shutil
import time
import json5
import torch
import numpy as np
from tqdm import tqdm
from utils.util import ValueWindow
from torch.utils.data import DataLoader
from models.tts.base.tts_trainer import TTSTrainer
from torch.nn import functional as F
from models.base.base_sampler import VariableSampler

from diffusers import get_scheduler

import accelerate
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from models.tts.vc.ns2_uniamphion import UniAmphionVC
from models.tts.vc.vc_dataset import  VCCollator, VCDataset, batch_by_size
# from models.tts.vc.vc_new_dataset import VCCollator, VCDataset, batch_by_size # used on ailab sever
from models.tts.vc.hubert_kmeans import HubertWithKmeans
from models.tts.vc.whisper_feature import WhisperNormal
from models.tts.vc.vc_loss import diff_loss, ConstractiveSpeakerLoss
from models.tts.vc.vc_utils import mel_spectrogram, extract_world_f0


class VCTrainer(TTSTrainer):
    def __init__(self, args, cfg):
        self.args = args
        self.cfg = cfg
        cfg.exp_name = args.exp_name
        self.content_extractor = "mhubert"

        # 初始化加速器，并确保所有进程都已就绪
        self._init_accelerator()
        self.accelerator.wait_for_everyone()

        # 在主进程中初始化日志记录器，避免在每个进程中重复记录
        if self.accelerator.is_main_process:
            self.logger = get_logger(args.exp_name, log_level="INFO")

        # 配置噪声和说话人使用
        self.use_source_noise = self.cfg.trans_exp.use_source_noise
        self.use_ref_noise = self.cfg.trans_exp.use_ref_noise
        self.use_speaker = self.cfg.trans_exp.use_speaker

        # 在主进程中记录配置信息
        if self.accelerator.is_main_process:
            self.logger.info(f"use_source_noise: {self.use_source_noise}")
            self.logger.info(f"use_ref_noise: {self.use_ref_noise}")
            self.logger.info(f"use_speaker: {self.use_speaker}")

        # 初始化一个时间窗口，用于监控或记录某些度量
        self.time_window = ValueWindow(50)

        # 记录训练开始信息
        if self.accelerator.is_main_process:
            self.logger.info("=" * 56)
            self.logger.info("||\t\tNew training process started.\t\t||")
            self.logger.info("=" * 56)
            self.logger.info("\n")
            self.logger.debug(f"Using {args.log_level.upper()} logging level.")
            self.logger.info(f"Experiment name: {args.exp_name}")
            self.logger.info(f"Experiment directory: {self.exp_dir}")


        # 初始化检查点目录，确保仅在主进程中执行
        self.checkpoint_dir = os.path.join(self.exp_dir, "checkpoint")
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self.logger.debug(f"Checkpoint directory: {self.checkpoint_dir}")

        # 初始化训练计数器
        self.batch_count: int = 0
        self.step: int = 0
        self.epoch: int = 0
        self.max_epoch = (self.cfg.train.max_epoch if self.cfg.train.max_epoch > 0 else float("inf"))
        if self.accelerator.is_main_process:
            self.logger.info(f"Max epoch: {self.max_epoch if self.max_epoch < float('inf') else 'Unlimited'}")

        # 检查基本配置
        if self.accelerator.is_main_process:
            self._check_basic_configs()
            self.save_checkpoint_stride = self.cfg.train.save_checkpoint_stride
            self.keep_last = [i if i > 0 else float("inf") for i in self.cfg.train.keep_last]
            self.run_eval = self.cfg.train.run_eval

        # 在所有进程中设置随机种子
        with self.accelerator.main_process_first():
            self._set_random_seed(self.cfg.train.random_seed)

        # setup data_loader
        with self.accelerator.main_process_first():
            if self.accelerator.is_main_process:
                self.logger.info("Building dataset...")
            self.train_dataloader = self._build_dataloader()
            self.speaker_num = len(self.train_dataloader.dataset.speaker2id)
            if self.accelerator.is_main_process:
                self.logger.info("Speaker num: {}".format(self.speaker_num))
            
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
        self.accelerator = accelerate.Accelerator(
            gradient_accumulation_steps=self.cfg.train.gradient_accumulation_step,
            log_with=self.cfg.train.tracker,
            project_config=project_config,
        )
        if self.accelerator.is_main_process:
            os.makedirs(project_config.project_dir, exist_ok=True)
            os.makedirs(project_config.logging_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        with self.accelerator.main_process_first():
            self.accelerator.init_trackers(self.args.exp_name)


    def _build_model(self):
        w2v  = HubertWithKmeans()
        self.cfg.model.vc_feature.content_feature_dim = 768
        model = UniAmphionVC(cfg=self.cfg.model, use_ref_noise=self.use_ref_noise, use_source_noise=self.use_source_noise)
        return model, w2v

    def _build_dataset(self):
        return VCDataset, VCCollator

    def _build_dataloader(self):
        np.random.seed(int(time.time()))
        if self.accelerator.is_main_process:
            self.logger.info("Use Dynamic Batchsize......")
        train_dataset = VCDataset(self.cfg.trans_exp)
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
        return train_loader

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
            for key, _ in model.items():
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
        device = self.accelerator.device 

        # 将所有Tensor类型的数据迁移到指定设备
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        speech = batch["speech"] 
        ref_speech = batch["ref_speech"] 
        
        with torch.set_grad_enabled(False):
            # 提取需要的特征和光谱图
            mel = mel_spectrogram(speech).transpose(1, 2)
            ref_mel = mel_spectrogram(ref_speech).transpose(1, 2)
            mask = batch["mask"]
            ref_mask = batch["ref_mask"]
            
            # 提取 pitch 和 content_feature
            if not self.use_source_noise:
                pitch = extract_world_f0(speech)
                pitch = (pitch - pitch.mean(dim=1, keepdim=True)) / (pitch.std(dim=1, keepdim=True) + 1e-6) # Normalize pitch (B,T)
                _, content_feature = self.w2v(speech) # semantic (B, T, 768)

            if self.use_ref_noise:
                noisy_ref_mel = mel_spectrogram(batch["noisy_ref_speech"]).transpose(1, 2)
                
            if self.use_source_noise:
                combined_speech = torch.cat((speech, batch["noisy_speech"]), dim=0)
                _, combined_features = self.w2v(combined_speech)
                content_feature, noisy_content_feature = torch.split(combined_features, speech.shape[0], dim=0)
                combined_pitch = extract_world_f0(combined_speech)
                clean_pitch, noisy_pitch = torch.split(combined_pitch, speech.shape[0], dim=0)
                pitch = (clean_pitch - clean_pitch.mean(dim=1, keepdim=True)) / (clean_pitch.std(dim=1, keepdim=True) + 1e-6)
                noisy_pitch = (noisy_pitch - noisy_pitch.mean(dim=1, keepdim=True)) / (noisy_pitch.std(dim=1, keepdim=True) + 1e-6)
        
        # FORWARD 模型
        if self.use_ref_noise and self.use_source_noise:
            diff_out, (ref_emb, noisy_ref_emb), (cond_emb, noisy_cond_emb) = self.model(
                x=mel, content_feature=content_feature, pitch=pitch, x_ref=ref_mel,
                x_mask=mask, x_ref_mask=ref_mask, noisy_x_ref=noisy_ref_mel,
                noisy_content_feature=noisy_content_feature, noisy_pitch=noisy_pitch
            )
        elif self.use_ref_noise:
            diff_out, (ref_emb, noisy_ref_emb), (cond_emb, _) = self.model(
                x=mel, content_feature=content_feature, pitch=pitch, x_ref=ref_mel,
                x_mask=mask, x_ref_mask=ref_mask, noisy_x_ref=noisy_ref_mel
            )
        else:
            diff_out, (ref_emb, _), (cond_emb, _) = self.model(
                x=mel, content_feature=content_feature, pitch=pitch, x_ref=ref_mel,
                x_mask=mask, x_ref_mask=ref_mask
            )

        if self.use_ref_noise:
            # B x N_query x D 
            ref_emb = torch.mean(ref_emb, dim=1) # B x D
            noisy_ref_emb = torch.mean(noisy_ref_emb, dim=1) # B x D
            all_ref_emb = torch.cat([ref_emb, noisy_ref_emb], dim=0) # 2B x D
            all_speaker_ids = torch.cat([batch["speaker_id"], batch["speaker_id"]], dim=0) # 2B
            cs_loss = self.contrastive_speaker_loss(all_ref_emb, all_speaker_ids) * 0.25
            total_loss += cs_loss
            train_losses["ref_loss"] = cs_loss

        if self.use_source_noise:
            # B x T x D
            diff_loss_cond = F.l1_loss(noisy_cond_emb, cond_emb, reduction="mean") * 2.0
            total_loss += diff_loss_cond
            train_losses["source_loss"] = diff_loss_cond

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

        train_losses['learning_rate'] = f"{self.optimizer.param_groups[0]['lr']:.1e}"
        train_losses["batch_size"] = batch["speaker_id"].shape[0]
        
        return (train_losses["total_loss"], train_losses, None)

    

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
        # Put the data to cuda device
        device = self.accelerator.device
        with device:
            torch.cuda.empty_cache()
        self.model = self.model.to(device)
        self.w2v = self.w2v.to(device)

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
            
            speech = batch["speech"].cpu().numpy()
            speech = speech[0]
            self.batch_count += 1
            self.step += 1
            #epoch_step += 1
            if len(speech) >= 16000 * 25:
                continue
            with self.accelerator.accumulate(self.model):
                total_loss, train_losses, _ = self._train_step(batch)
            
            if self.batch_count % self.cfg.train.gradient_accumulation_step == 0:
                epoch_sum_loss += total_loss
                self.current_loss = total_loss
                if isinstance(train_losses, dict):
                    for key, loss in train_losses.items():
                        self.accelerator.log(
                            {"Epoch/Train {} Loss".format(key): loss},
                            step=self.step,
                        )
                if (self.accelerator.is_main_process and self.batch_count % 10 == 0):
                    self.echo_log(train_losses, mode="Training")
                
                self.save_checkpoint()
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
                self.logger.info("Start training......")
            
            train_total_loss, _ = self._train_epoch()

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
        if self.accelerator.is_main_process:
            self.logger.info("Training finished......")

    def save_checkpoint(self):
        self.accelerator.wait_for_everyone()
        # main process only
        if self.accelerator.is_main_process:
            if self.batch_count % self.save_checkpoint_stride[0] == 0:
                keep_last = self.keep_last[0]
                # 读取self.checkpoint_dir所有的folder
                all_ckpts = os.listdir(self.checkpoint_dir)
                # 排除非文件夹
                all_ckpts = [ckpt for ckpt in all_ckpts if os.path.isdir(os.path.join(self.checkpoint_dir, ckpt))]
                if len(all_ckpts) > keep_last:
                    # 只保留keep_last个的folder in self.checkpoint_dir, sort by step  "epoch-{:04d}_step-{:07d}_loss-{:.6f}"
                    all_ckpts = sorted(all_ckpts, key=lambda x: int(x.split("_")[1].split('-')[1]))

                    # 只保留keep_last个的folder in self.checkpoint_dir, sort by step  "epoch-{:04d}_step-{:07d}_loss-{:.6f}"
                    all_ckpts = sorted(all_ckpts, key=lambda x: int(x.split("_")[1].split('-')[1]))
                    for ckpt in all_ckpts[:-keep_last]:
                        shutil.rmtree(os.path.join(self.checkpoint_dir, ckpt))
                checkpoint_filename = "epoch-{:04d}_step-{:07d}_loss-{:.6f}".format(
                    self.epoch, self.step, self.current_loss
                )
                path = os.path.join(self.checkpoint_dir, checkpoint_filename)
                self.logger.info("Saving state to {}...".format(path))
                self.accelerator.save_state(path)
                self.logger.info("Finished saving state.")
        self.accelerator.wait_for_everyone()
