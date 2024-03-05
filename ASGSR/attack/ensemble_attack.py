import torch
from torch.utils.data import DataLoader
import os
import shutil

import sys

sys.path.append('../')
sys.path.append('./')
import dataset
import attacks
from ensemble_attack_config import config
from utils.shell_info import read_num_workers
from utils.logger import create_logger
from datetime import datetime
from utils.io_utils import load_voxcelebtrainer_model, load_yaml_to_dict, save_waveform_torch
from attacks.wrappers.multiattack import MultiAttack
from voxceleb_trainer.tuneThreshold import ComputeEqualErrorRate

if __name__ == '__main__':
    # save_dir
    attack_names = config.attack.names
    # The ensemble attack may contain many models,
    # and using the model and attack method as the experiment ID (unique key) may have too long folder name,
    # so the experiment time is used as the unique ID.
    exp_mark = config.attack.mark  # artificially defined experimental markers
    exp_id = '-'.join(attack_names) + '-' + exp_mark + '-' + datetime.now().strftime('%Y%m%d%H%M%S')

    save_dir = os.path.join(
        'ensemble_attack_exps',
        config.data.dataset_name,
        exp_id
    )  # 'ensamble_attack_exps/VoxCeleb1Verification/PGD_ResNetSE34V2-PGD_ECAPATDNN-PGD_RawNet3-20230724234147'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # avder dir
    adv_dir = os.path.join(
        config.adv_sample_root,
        config.data.dataset_name,
        exp_id
    )  # /mntcephfs/lab_data/wangli/ASGSR/ensemble/VoxCeleb1Verification/PGD_ResNetSE34V2-PGD_ECAPATDNN-PGD_RawNet3-20230724234147
    if not os.path.exists(adv_dir):
        os.makedirs(adv_dir)

    # logger
    logger = create_logger(__name__, save_dir)
    logger.info('save dir: {}'.format(save_dir))
    logger.info('adver dir: {}'.format(adv_dir))

    # gpu
    num_gpu = torch.cuda.device_count()
    if num_gpu == 0:
        logger.error('no gpu available')
        exit(1)
    if num_gpu > 1:
        logger.error('only support single gpu')
        exit(1)
    device = torch.device('cuda:0')

    # attackers
    attackers = []
    for attacker_name in config.attack.names:
        attacker_config = config.attack[attacker_name]
        model_name = attacker_config.model.name
        model_config_path = attacker_config.model.config_path
        model_config = load_yaml_to_dict(model_config_path)
        model_param_path = attacker_config.model.save_path
        model = load_voxcelebtrainer_model(model_name, model_config, model_param_path)
        model.threshold = attacker_config.model.threshold
        model = model.to(device)
        model.eval()

        attack_name = attacker_config.attack.name
        attack_config = attacker_config.attack.config
        attacker = getattr(attacks, attack_name)(model, device=device, **attack_config)

        logger.info('load attacker: {}'.format(attacker_name))
        attackers.append(attacker)
    multi_attacker = MultiAttack(attackers, **config.attack)

    # num_workers
    sh_file_path = config.sh_file_path
    num_workers = read_num_workers(sh_file_path)

    # data
    dataset_name = config.data.dataset_name
    dataset_config = config.data[dataset_name].dataset
    dataloader_config = config.data[dataset_name].dataloader
    dataset = getattr(dataset, dataset_name)(**dataset_config)
    dataloader = DataLoader(dataset, num_workers=num_workers, **dataloader_config)
    logger.info('dataset: {}'.format(dataset_name))

    # max_iter = 10， 100%攻击率
    # max_iter 迁移攻击，不一定越大越好
    # 攻击成功相关的参数：epsilon（设置为list [0.001, 0.05]，对比曲线），
    # epsilon 白盒： [0.001, 0.005]
    # epsilon 黑盒： [0.01, 0.05]
    # step_size
    #     1.固定值(每步添加的最大噪声幅度)，
    #     2.step_size=(epsilon/max_iter)*2
    # batch_size设置为1，兼容更多模型
    # EOT设置为1

    # attack
    target_success_cnt = 0
    untarget_success_cnt = 0
    target_cnt = 0
    untarget_cnt = 0
    perturbation = 0
    all_scores = []  # for EER
    all_labels = []
    result_file_path = os.path.join(save_dir, 'attackResult.txt')
    for item in dataloader:
        enroll_waveforms = item[
            config.data[dataset_name].enroll_waveform_index]  # shape: (batch_size, channels, audio_len)
        eval_waveforms = item[config.data[dataset_name].eval_waveform_index]
        sample_rate = item[config.data[dataset_name].sample_rate_index]
        labels = item[config.data[dataset_name].label_index]
        enroll_files = item[config.data[dataset_name].enroll_file_index]
        eval_files = item[config.data[dataset_name].eval_file_index]
        enroll_speakers = item[config.data[dataset_name].enroll_speaker_index]
        eval_speakers = item[config.data[dataset_name].eval_speaker_index]

        enroll_waveforms, eval_waveforms, labels = enroll_waveforms.to(device), eval_waveforms.to(device), labels.to(
            device)

        adv_waveforms, is_successes, similarity_scores, average_pertubations = multi_attacker(enroll_waveforms,
                                                                                              eval_waveforms,
                                                                                              labels)  # [batch, C, L]

        # record adversarial results and save adversarial audios
        result_file = open(result_file_path, mode='a+')
        is_successes = is_successes.detach().cpu()
        adv_waveforms = adv_waveforms.detach().cpu()
        labels = labels.detach().cpu()
        all_scores.append(similarity_scores.detach().cpu())
        similarity_scores = similarity_scores.detach().cpu().mean(dim=1)  # recode all attack average similarity score

        for adv_waveform, is_success, enroll_file, eval_file, label, similarity_score, average_pertubation in zip(
                adv_waveforms, is_successes,
                enroll_files, eval_files, labels,
                similarity_scores, average_pertubations):
            adv_file_name = '{}_{}'.format(enroll_file, eval_file)
            adv_file_path = os.path.join(adv_dir, adv_file_name + ".wav")
            save_waveform_torch(adv_file_path, adv_waveform, sample_rate)
            result_file.write(
                '{} {} {} {} {} {}\n'.format(enroll_file, adv_file_name, is_success, label, similarity_score.item(),
                                             average_pertubation.item()))
            if label == 1:
                untarget_cnt += 1
                untarget_success_cnt += is_success.item()
            else:  # label == 0
                target_cnt += 1
                target_success_cnt += is_success.item()
            all_labels.append(label.item())
            perturbation += average_pertubation.item()
        result_file.close()

        success_rate = (target_success_cnt + untarget_success_cnt) / (target_cnt + untarget_cnt)
        target_success_rate = 0 if target_cnt == 0 else target_success_cnt / target_cnt
        untarget_success_rate = 0 if untarget_cnt == 0 else untarget_success_cnt / untarget_cnt
        logger.info(
            'success rate: {}/{}={}'.format(target_success_cnt + untarget_success_cnt, target_cnt + untarget_cnt,
                                            success_rate))
        logger.info('target success rate: {}/{}={}'.format(target_success_cnt, target_cnt, target_success_rate))
        logger.info('untarget success rate: {}/{}={}'.format(untarget_success_cnt, untarget_cnt, untarget_success_rate))

    # EER
    all_scores = torch.cat(all_scores, dim=0)
    all_scores = all_scores.T
    all_scores = all_scores.tolist()
    for attacker_name, scores in zip(config.attack.names, all_scores):
        EER = ComputeEqualErrorRate(scores, all_labels)
        logger.info('{} EER: {}'.format(attacker_name, EER))
    logger.info('Perturbation: {}'.format(perturbation / (target_cnt + untarget_cnt)))
    logger.info('arrange attack result: success rate: {}, target success rate: {}, untarget success rate {}'.format(
        round(success_rate * 100, 1), round(target_success_rate * 100, 1), round(untarget_success_rate * 100, 1)))

    # save scrips
    shutil.copy(sh_file_path, save_dir)
    shutil.copy('ensemble_attack.py', save_dir)
    shutil.copy('ensemble_attack_config.py', save_dir)
    shutil.copy(result_file_path, adv_dir)
