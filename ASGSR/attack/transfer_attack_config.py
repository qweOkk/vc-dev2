from easydict import EasyDict as edict

config = edict()
config.sh_file_path = '/home/wangli/ASGSR/attack/transferattackjob.sh'

# ---------------------------------------- attack --------------------------------------- #
config.attack = edict()

## digital attack, until success
# config.attack.attack_sample_dirs = [
#     '/mntcephfs/lab_data/wangli/ASGSR/attack/VoxCeleb1Verification/attack/PGD_ECAPATDNN-ECAPATDNN_eps-0.004_alpha-0.0004_until_success-True',
#     '/mntcephfs/lab_data/wangli/ASGSR/attack/VoxCeleb1Verification/attack/PGD_RawNet3-model_eps-0.004_alpha-0.0004_until_success-True',
#     '/mntcephfs/lab_data/wangli/ASGSR/attack/VoxCeleb1Verification/attack/PGD_ResNetSE34V2-ResNetSE34V2_eps-0.004_alpha-0.0004_until_success-True',
#     '/mntcephfs/lab_data/wangli/ASGSR/attack/VoxCeleb1Verification/attack/PGD_XVector-model000000230_eps-0.004_alpha-0.0004_until_success-True'
# ]

## digital attack, 20 steps
# config.attack.attack_sample_dirs = [
#     '/mntcephfs/lab_data/wangli/ASGSR/attack/VoxCeleb1Verification/attack/PGD_ECAPATDNN-ECAPATDNN_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/attack/VoxCeleb1Verification/attack/PGD_RawNet3-model_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/attack/VoxCeleb1Verification/attack/PGD_ResNetSE34V2-ResNetSE34V2_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/attack/VoxCeleb1Verification/attack/PGD_XVector-model000000230_eps-0.008_alpha-0.0004_steps-20'
# ]

## Every PGD attack until success, until ensemble attack success
# config.attack.attack_sample_dirs = [
#     '/mntcephfs/lab_data/wangli/ASGSR/ensemble/VoxCeleb1Verification/PGD_XVector-PGD_ResNetSE34V2-PGD_ECAPATDNN-20230724201833'
#     '/mntcephfs/lab_data/wangli/ASGSR/ensemble/VoxCeleb1Verification/PGD_XVector-PGD_ECAPATDNN-PGD_RawNet3-20230724212909',
#     '/mntcephfs/lab_data/wangli/ASGSR/ensemble/VoxCeleb1Verification/PGD_XVector-PGD_ResNetSE34V2-PGD_RawNet3-20230724212852',
#     '/mntcephfs/lab_data/wangli/ASGSR/ensemble/VoxCeleb1Verification/PGD_ResNetSE34V2-PGD_ECAPATDNN-PGD_RawNet3-20230724234147'
# ]

## Every PGD attack 20 step, until ensemble attack success
# config.attack.attack_sample_dirs = [
#     '/mntcephfs/lab_data/wangli/ASGSR/ensemble/VoxCeleb1Verification/PGD_XVector-PGD_ResNetSE34V2-PGD_RawNet3-Step20-20230727171220',
#     '/mntcephfs/lab_data/wangli/ASGSR/ensemble/VoxCeleb1Verification/PGD_XVector-PGD_ResNetSE34V2-PGD_ECAPATDNN-Step20-20230727171240',
#     '/mntcephfs/lab_data/wangli/ASGSR/ensemble/VoxCeleb1Verification/PGD_XVector-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727171124',
#     '/mntcephfs/lab_data/wangli/ASGSR/ensemble/VoxCeleb1Verification/PGD_ResNetSE34V2-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727170536'
# ]

## record, PGD attack 20 step, until ensemble attack success, sound x
# config.attack.attack_sample_dirs = [
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/iphone/iphone_PGD_ECAPATDNN-ECAPATDNN_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/iphone/iphone_PGD_RawNet3-model_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/iphone/iphone_PGD_XVector-model000000230_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/iphone/iphone_PGD_ResNetSE34V2-ResNetSE34V2_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/iphone/iphone_PGD_ResNetSE34V2-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727170536',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/iphone/iphone_PGD_XVector-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727171124',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/iphone/iphone_PGD_XVector-PGD_ResNetSE34V2-PGD_ECAPATDNN-Step20-20230727171240',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/iphone/iphone_PGD_XVector-PGD_ResNetSE34V2-PGD_RawNet3-Step20-20230727171220'
# ]
#
# config.attack.attack_sample_dirs = [
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/mate/mate_PGD_ECAPATDNN-ECAPATDNN_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/mate/mate_PGD_RawNet3-model_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/mate/mate_PGD_XVector-model000000230_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/mate/mate_PGD_ResNetSE34V2-ResNetSE34V2_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/mate/mate_PGD_ResNetSE34V2-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727170536',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/mate/mate_PGD_XVector-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727171124',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/mate/mate_PGD_XVector-PGD_ResNetSE34V2-PGD_ECAPATDNN-Step20-20230727171240',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/mate/mate_PGD_XVector-PGD_ResNetSE34V2-PGD_RawNet3-Step20-20230727171220'
# ]
#
# config.attack.attack_sample_dirs = [
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/k40/k40_PGD_ECAPATDNN-ECAPATDNN_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/k40/k40_PGD_RawNet3-model_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/k40/k40_PGD_XVector-model000000230_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/k40/k40_PGD_ResNetSE34V2-ResNetSE34V2_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/k40/k40_PGD_ResNetSE34V2-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727170536',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/k40/k40_PGD_XVector-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727171124',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/k40/k40_PGD_XVector-PGD_ResNetSE34V2-PGD_ECAPATDNN-Step20-20230727171240',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/attack/k40/k40_PGD_XVector-PGD_ResNetSE34V2-PGD_RawNet3-Step20-20230727171220'
# ]

# # record, PGD attack 20 step, until ensemble attack success, philip
# config.attack.attack_sample_dirs = [
#     # iphone
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/iphone/iphone_PGD_ECAPATDNN-ECAPATDNN_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/iphone/iphone_PGD_RawNet3-model_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/iphone/iphone_PGD_XVector-model000000230_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/iphone/iphone_PGD_ResNetSE34V2-ResNetSE34V2_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/iphone/iphone_PGD_ResNetSE34V2-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727170536',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/iphone/iphone_PGD_XVector-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727171124',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/iphone/iphone_PGD_XVector-PGD_ResNetSE34V2-PGD_ECAPATDNN-Step20-20230727171240',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/iphone/iphone_PGD_XVector-PGD_ResNetSE34V2-PGD_RawNet3-Step20-20230727171220',
#     # mate50
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/mate50/mate50_PGD_ECAPATDNN-ECAPATDNN_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/mate50/mate50_PGD_RawNet3-model_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/mate50/mate50_PGD_XVector-model000000230_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/mate50/mate50_PGD_ResNetSE34V2-ResNetSE34V2_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/mate50/mate50_PGD_ResNetSE34V2-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727170536',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/mate50/mate50_PGD_XVector-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727171124',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/mate50/mate50_PGD_XVector-PGD_ResNetSE34V2-PGD_ECAPATDNN-Step20-20230727171240',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/mate50/mate50_PGD_XVector-PGD_ResNetSE34V2-PGD_RawNet3-Step20-20230727171220',
#     # k40
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/k40/k40_PGD_ECAPATDNN-ECAPATDNN_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/k40/k40_PGD_RawNet3-model_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/k40/k40_PGD_XVector-model000000230_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/k40/k40_PGD_ResNetSE34V2-ResNetSE34V2_eps-0.008_alpha-0.0004_steps-20',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/k40/k40_PGD_ResNetSE34V2-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727170536',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/k40/k40_PGD_XVector-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727171124',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/k40/k40_PGD_XVector-PGD_ResNetSE34V2-PGD_ECAPATDNN-Step20-20230727171240',
#     '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/philip/k40/k40_PGD_XVector-PGD_ResNetSE34V2-PGD_RawNet3-Step20-20230727171220'
# ]

# record, PGD attack 20 step, until ensemble attack success, flip
config.attack.attack_sample_dirs = [
    # iphone
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/iphone/iphone_PGD_ECAPATDNN-ECAPATDNN_eps-0.008_alpha-0.0004_steps-20',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/iphone/iphone_PGD_RawNet3-model_eps-0.008_alpha-0.0004_steps-20',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/iphone/iphone_PGD_XVector-model000000230_eps-0.008_alpha-0.0004_steps-20',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/iphone/iphone_PGD_ResNetSE34V2-ResNetSE34V2_eps-0.008_alpha-0.0004_steps-20',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/iphone/iphone_PGD_ResNetSE34V2-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727170536',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/iphone/iphone_PGD_XVector-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727171124',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/iphone/iphone_PGD_XVector-PGD_ResNetSE34V2-PGD_ECAPATDNN-Step20-20230727171240',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/iphone/iphone_PGD_XVector-PGD_ResNetSE34V2-PGD_RawNet3-Step20-20230727171220',
    # mate50
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/mate50/mate50_PGD_ECAPATDNN-ECAPATDNN_eps-0.008_alpha-0.0004_steps-20',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/mate50/mate50_PGD_RawNet3-model_eps-0.008_alpha-0.0004_steps-20',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/mate50/mate50_PGD_XVector-model000000230_eps-0.008_alpha-0.0004_steps-20',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/mate50/mate50_PGD_ResNetSE34V2-ResNetSE34V2_eps-0.008_alpha-0.0004_steps-20',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/mate50/mate50_PGD_ResNetSE34V2-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727170536',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/mate50/mate50_PGD_XVector-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727171124',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/mate50/mate50_PGD_XVector-PGD_ResNetSE34V2-PGD_ECAPATDNN-Step20-20230727171240',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/mate50/mate50_PGD_XVector-PGD_ResNetSE34V2-PGD_RawNet3-Step20-20230727171220',
    # k40
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/k40/k40_PGD_ECAPATDNN-ECAPATDNN_eps-0.008_alpha-0.0004_steps-20',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/k40/k40_PGD_RawNet3-model_eps-0.008_alpha-0.0004_steps-20',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/k40/k40_PGD_XVector-model000000230_eps-0.008_alpha-0.0004_steps-20',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/k40/k40_PGD_ResNetSE34V2-ResNetSE34V2_eps-0.008_alpha-0.0004_steps-20',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/k40/k40_PGD_ResNetSE34V2-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727170536',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/k40/k40_PGD_XVector-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727171124',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/k40/k40_PGD_XVector-PGD_ResNetSE34V2-PGD_ECAPATDNN-Step20-20230727171240',
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification/flip/k40/k40_PGD_XVector-PGD_ResNetSE34V2-PGD_RawNet3-Step20-20230727171220'
]

config.attack.defense_models = [
    'ECAPATDNN',
    'RawNet3',
    'XVector',
    'ResNetSE34V2',
]

config.attack.score_function = edict()
config.attack.score_function.name = 'cosine_similarity_score'
config.attack.score_function.cosine_similarity_score = edict()
config.attack.decision_function = edict()
config.attack.decision_function.name = 'cosine_similarity_decision'
config.attack.decision_function.cosine_similarity_decision = edict()

# ---------------------------------------- model ---------------------------------------- #
config.model = edict()

# ResNetSE34V2
config.model.ResNetSE34V2 = edict()
config.model.ResNetSE34V2.save_path = '/home/wangli/ASGSR/pretrained_models/ResNetSE34V2/ResNetSE34V2.model'
config.model.ResNetSE34V2.config_path = '/home/wangli/ASGSR/voxceleb_trainer/configs/ResNetSE34V2.yaml'
config.model.ResNetSE34V2.threshold = 0.3347

# ECAPATDNN
config.model.ECAPATDNN = edict()
config.model.ECAPATDNN.save_path = '/home/wangli/ASGSR/pretrained_models/ECAPATDNN/ECAPATDNN.pth'
config.model.ECAPATDNN.config_path = '/home/wangli/ASGSR/voxceleb_trainer/configs/ECAPATDNN.yaml'
config.model.ECAPATDNN.threshold = 0.27144

# RawNet3
config.model.RawNet3 = edict()
config.model.RawNet3.save_path = '/home/wangli/ASGSR/pretrained_models/RawNet3/model.model'
config.model.RawNet3.config_path = '/home/wangli/ASGSR/voxceleb_trainer/configs/RawNet3_AAM.yaml'
config.model.RawNet3.threshold = 0.29348

# XVEC
config.model.XVector = edict()
config.model.XVector.save_path = '/home/wangli/ASGSR/voxceleb_trainer/exps/XVector_AAM/model/model000000230.model'
config.model.XVector.config_path = '/home/wangli/ASGSR/voxceleb_trainer/configs/XVector_AAM.yaml'
config.model.XVector.threshold = 0.28011

# ------------------------------------------ data ----------------------------------------#

# data
config.data = edict()
config.data.dataset_name = 'VoxCeleb1VerificationAttack'

# VoxCeleb1VerificationTest
config.data.VoxCeleb1VerificationAttack = edict()
config.data.VoxCeleb1VerificationAttack.dataset = edict()
config.data.VoxCeleb1VerificationAttack.dataloader = edict()
config.data.VoxCeleb1VerificationAttack.dataset.voxceleb1_file_dir = '/mntcephfs/data/chenxi/datasets/VoxCeleb/VoxCeleb1/wav'
config.data.VoxCeleb1VerificationAttack.dataloader.batch_size = 1
config.data.VoxCeleb1VerificationAttack.enroll_waveform_index = 0
config.data.VoxCeleb1VerificationAttack.eval_waveform_index = 1
config.data.VoxCeleb1VerificationAttack.enroll_file_id_index = 2
config.data.VoxCeleb1VerificationAttack.eval_file_id_index = 3
config.data.VoxCeleb1VerificationAttack.label_index = 4
config.data.VoxCeleb1VerificationAttack.is_ori_success_index = 5

config.data.ASVspoof2019 = edict()
config.data.ASVspoof2019.dataset = edict()
config.data.ASVspoof2019.dataset.data_file = '/home/wangli/ASGSR/audio_record/enroll_eval_pairs.txt'
config.data.ASVspoof2019.dataset.train_path = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac'
config.data.ASVspoof2019.dataset.dev_path = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_dev/flac'
config.data.ASVspoof2019.dataset.eval_path = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_eval/flac'
config.data.ASVspoof2019.waveform_index_spk1 = 0
config.data.ASVspoof2019.waveform_index_spk2 = 1
config.data.ASVspoof2019.label_index = 3
config.data.ASVspoof2019.enroll_file_index = 4
config.data.ASVspoof2019.test_file_index = 5  # spk1 for enroll, spk2 for test
