import torch
import os
from torch.utils.data import DataLoader

import sys

sys.path.append('./')
sys.path.append('../')
# import dataset
from utils.logger import create_logger
from utils.shell_info import read_num_workers
from utils.io_utils import load_voxcelebtrainer_model, load_yaml_to_dict
from attacks import score_functions
from attacks import decision_functions
from voxceleb_trainer.tuneThreshold import ComputeEqualErrorRate
from transfer_attack_config import config


def transfer_attack(num_gpu, gpu_id, attack_sample_dir, defense_model_name, config):
    import dataset
    # save dir
    if attack_sample_dir[-1] == '/':
        attack_sample_dir = attack_sample_dir[:-1]  # not include '/'
    attack_info = attack_sample_dir.split('/')[-1]  # PGD_XVEC-20230305192215_eps-0.001
    defence_model_config_path = config.model[defense_model_name].config_path
    defence_model_config = load_yaml_to_dict(defence_model_config_path)
    defense_model_param_path = config.model[defense_model_name].save_path
    defence_model_info = '{}-{}'.format(defense_model_name, defense_model_param_path.split('/')[-1].split('.')[0])
    save_dir = os.path.join(
        'transfer_attack_exps',
        config.data.dataset_name,
        'flip',
        '{}_VS_{}'.format(attack_info, defence_model_info)
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # logger
    logger = create_logger(__name__, save_dir)
    logger.info('save dir: {}'.format(save_dir))

    logger.info('attack: {}'.format(attack_sample_dir))
    logger.info('defense model: {}'.format(defence_model_info))

    # gpu device
    gpu_id = gpu_id % num_gpu
    device = torch.device('cuda:{}'.format(gpu_id))

    # defense model
    defence_model = load_voxcelebtrainer_model(defense_model_name, defence_model_config, defense_model_param_path)
    # defence_model.threshold = config.model[defense_model_name].threshold
    defence_model = defence_model.to(device)
    defence_model.eval()
    logger.info('load defense model from: {}'.format(defense_model_param_path))

    # num workers
    sh_file_path = config.sh_file_path
    num_workers = read_num_workers(sh_file_path) // num_gpu

    # dataset & dataloader
    dataset_name = config.data.dataset_name
    dataset_config = config.data[dataset_name].dataset
    dataloader_config = config.data[dataset_name].dataloader
    attack_result_file = os.path.join(attack_sample_dir, 'attackResult.txt')
    dataset = getattr(dataset, dataset_name)(attack_result_file=attack_result_file, attack_file_dir=attack_sample_dir,
                                             **dataset_config)
    dataloader = DataLoader(dataset, num_workers=num_workers, **dataloader_config)

    # score & decision function
    score_function_name = config.attack.score_function.name
    score_function_config = config.attack.score_function[score_function_name]
    score_funciton = getattr(score_functions, score_function_name)(score_function_config)
    decision_function_name = config.attack.decision_function.name
    decision_function_config = config.attack.decision_function[decision_function_name]
    decision_function = getattr(decision_functions, decision_function_name)(decision_function_config,
                                                                            threshold=config.model[
                                                                                defense_model_name].threshold)

    # transfer attack
    target_success_cnt = 0
    untarget_success_cnt = 0
    target_cnt = 0
    untarget_cnt = 0
    all_scores = []  # for EER
    all_labels = []
    for item in dataloader:
        transfer_attack_result_file = open(os.path.join(save_dir, 'transfer_attack_result.txt'), mode='a+')
        enroll_waveforms = item[config.data[dataset_name].enroll_waveform_index]
        eval_waveforms = item[config.data[dataset_name].eval_waveform_index]
        enroll_file_ids = item[config.data[dataset_name].enroll_file_id_index]
        eval_file_ids = item[config.data[dataset_name].eval_file_id_index]
        labels = item[config.data[dataset_name].label_index]
        is_ori_successes = item[config.data[dataset_name].is_ori_success_index]

        enroll_waveforms = enroll_waveforms.to(device)
        eval_waveforms = eval_waveforms.to(device)
        labels = labels.to(device)

        enroll_embeddings = defence_model(enroll_waveforms)
        eval_embeddings = defence_model(eval_waveforms)

        similarity_scores = score_funciton(enroll_embeddings, eval_embeddings)
        decisions = decision_function(enroll_embeddings, eval_embeddings)
        is_transfer_successes = torch.logical_xor(decisions.bool(), labels.bool())

        for enroll_file_id, eval_file_id, is_ori_success, is_transfer_success, label, similarity_score in zip(
                enroll_file_ids, eval_file_ids,
                is_ori_successes,
                is_transfer_successes,
                labels, similarity_scores):
            transfer_attack_result_file.write(
                '{} {} {} {} {} {}\n'.format(enroll_file_id, eval_file_id, is_ori_success.item(),
                                             is_transfer_success.item(), label.item(),
                                             similarity_score.item()))
            if label == 1:
                untarget_cnt += 1
                if is_transfer_success == 1:
                    untarget_success_cnt += 1
            else:
                target_cnt += 1
                if is_transfer_success == 1:
                    target_success_cnt += 1
            all_scores.append(similarity_score.item())
            all_labels.append(label.item())
        transfer_attack_result_file.close()

        success_rate = (target_success_cnt + untarget_success_cnt) / (target_cnt + untarget_cnt)

        target_success_rate = 0 if target_cnt == 0 else target_success_cnt / target_cnt
        untarget_success_rate = 0 if untarget_cnt == 0 else untarget_success_cnt / untarget_cnt
        logger.info(
            'success rate: {}/{}={}'.format(target_success_cnt + untarget_success_cnt, target_cnt + untarget_cnt,
                                            success_rate))
        logger.info('target success rate: {}/{}={}'.format(target_success_cnt, target_cnt, target_success_rate))
        logger.info('untarget success rate: {}/{}={}'.format(untarget_success_cnt, untarget_cnt, untarget_success_rate))

    if target_cnt > 0 and untarget_cnt > 0:
        eer = ComputeEqualErrorRate(all_scores, all_labels)
        logger.info('EER: {}'.format(eer))
    else:
        logger.info('target count: {}, untarget count: {}, EER: None'.format(target_cnt, untarget_cnt))
    logger.info('arrange attack result: success rate: {}, target success rate: {}, untarget success rate {}'.format(
        round(success_rate * 100, 1), round(target_success_rate * 100, 1), round(untarget_success_rate * 100, 1)))
    logger.handlers.clear()


if __name__ == '__main__':
    num_gpu = torch.cuda.device_count()
    if num_gpu == 0:
        print('No GPU available!')
        exit(1)
    gpu_id = 0
    # pool = Pool(processes=num_gpu * 4)
    for attack_sample_dir in config.attack.attack_sample_dirs:
        for defense_model_name in config.attack.defense_models:
            transfer_attack(num_gpu, gpu_id, attack_sample_dir, defense_model_name, config)
            gpu_id = (gpu_id + 1) % num_gpu
            # pool.apply_async(transfer_attack, args=(num_gpu, gpu_id, attack_sample_dir, defense_model_name, config))
    # pool.close()
    # pool.join()
