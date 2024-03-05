import sys
import numpy as np

sys.path.append('../')
sys.path.append('.')
from utils.io_utils import save_waveform_torch, load_waveform_torch, load_voxcelebtrainer_model, load_yaml_to_dict
import torch
from attack_config import config as attack_config
from ensemble_attack_config import config as ensemble_attack_config
import attacks
from attacks.wrappers.multiattack import MultiAttack

# device
device = torch.device('cuda:0')

# # model
# model_name = attack_config.model.model_name
# model_config_path = attack_config.model[model_name].config_path
# model_config = load_yaml_to_dict(model_config_path)
# model_param_path = attack_config.model[model_name].save_path
# model = load_voxcelebtrainer_model(model_name, model_config, model_param_path)
# model.threshold = attack_config.model[model_name].threshold
# model = model.to(device)
# model.eval()
#
# # attacker
# attacker_name = attack_config.attack.attack_name
# attacker_config = attack_config.attack[attacker_name]
# attacker = getattr(attacks, attacker_name)(model, device=device, **attacker_config)


# # attack
# enroll_file_path = '/mntcephfs/data/chenxi/datasets/VoxCeleb/VoxCeleb1/wav/id10270/8jEAjG6SegY/00035.wav'
# enroll_waveform, _ = load_waveform_torch(enroll_file_path)
# enroll_waveform = enroll_waveform.to(device)
# enroll_waveform = enroll_waveform.unsqueeze(0)
#
# # zero
# # silence_waveform = torch.zeros(1, 1, enroll_waveform.size()[2]).to(device)
# # labels = torch.tensor([0]).to(device)
# # adv_waveforms, is_successes, similarity_scores, average_pertubations = attacker(enroll_waveform, silence_waveform,
# #                                                                                 labels)  # [batch, C, L]
# # print(is_successes)
# # print(attacker.display_info())
# #
# # adv_waveforms = adv_waveforms.detach().cpu()
# # save_waveform_torch('/home/wangli/ASGSR/attack/test/zero_audio.wav', adv_waveforms[0], 16000)
#
# # 440hz
# sample_rate = 44100  # 采样率，每秒采样次数
# frequency = 440.0  # 正弦波频率，440 Hz 表示 A4 音高
# duration = enroll_waveform.size()[2] / 16000  # 音频时长，单位秒
# t = np.linspace(0, duration, int(sample_rate * duration), False)
# sin_wave = 0.5 * np.sin(2 * np.pi * frequency * t)
# sin_wave_tensor = torch.tensor(sin_wave, dtype=torch.float32).view(1, 1, -1)
# sin_wave_tensor = sin_wave_tensor.to(device)
# labels = torch.tensor([0]).to(device)
#
# adv_waveforms, is_successes, similarity_scores, average_pertubations = attacker(enroll_waveform, sin_wave_tensor,
#                                                                                 labels)
# adv_waveforms = adv_waveforms.detach().cpu()
# # save_waveform_torch('/home/wangli/ASGSR/attack/test/440sin_audio.wav', adv_waveforms[0], 16000)
# print(is_successes)
# print(attacker.display_info())


# ensemble attacker
attackers = []
for attacker_name in ensemble_attack_config.attack.names:
    attacker_config = ensemble_attack_config.attack[attacker_name]
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

    attackers.append(attacker)
multi_attacker = MultiAttack(attackers, **ensemble_attack_config.attack)

enroll_file_path = '/home/wangli/WechatAttack/record1.wav'
enroll_waveform, _ = load_waveform_torch(enroll_file_path)
enroll_waveform = enroll_waveform.to(device)
enroll_waveform = enroll_waveform.unsqueeze(0)
eval_file_path = '/home/wangli/ASGSR/attack/test/1697121918744.wav'
eval_waveform, _ = load_waveform_torch(eval_file_path)
eval_waveform = eval_waveform.to(device)
eval_waveform = eval_waveform.unsqueeze(0)
labels = torch.tensor([0]).to(device)

adv_waveforms, is_successes, similarity_scores, average_pertubations = multi_attacker(enroll_waveform,
                                                                                      eval_waveform,
                                                                                      labels)

adv_waveforms = adv_waveforms.detach().cpu()
save_waveform_torch('/home/wangli/ASGSR/attack/test/record1_1697121918744.wav', adv_waveforms[0], 16000)

print(is_successes)
