from easydict import EasyDict as edict

config = edict()
config.sh_file_path = '/home/wangli/ASGSR/attack/attackjob.sh'
config.adv_sample_root = '/mntcephfs/lab_data/wangli/ASGSR/attack'

# ---------------------------------------- dataset --------------------------------------- #
config.data = edict()
config.data.dataset_name = 'VoxCeleb1Verification'

# VoxCeleb1Verification
config.data.VoxCeleb1Verification = edict()
config.data.VoxCeleb1Verification.dataset = edict()
config.data.VoxCeleb1Verification.dataloader = edict()
config.data.VoxCeleb1Verification.dataset.root = '/mntcephfs/data/chenxi/datasets/VoxCeleb/VoxCeleb1/wav'
# config.data.VoxCeleb1Verification.dataset.meta_file = '/mntcephfs/data/chenxi/datasets/VoxCeleb/VoxCeleb1/veri_test.txt'
config.data.VoxCeleb1Verification.dataset.meta_file = '/home/wangli/ASGSR/utils/vox1_uniform_sample_25.txt'
config.data.VoxCeleb1Verification.dataloader.batch_size = 1
config.data.VoxCeleb1Verification.enroll_waveform_index = 0
config.data.VoxCeleb1Verification.eval_waveform_index = 1
config.data.VoxCeleb1Verification.sample_rate_index = 2
config.data.VoxCeleb1Verification.label_index = 3
config.data.VoxCeleb1Verification.enroll_file_index = 4
config.data.VoxCeleb1Verification.eval_file_index = 5  # spk1 for enroll, spk2 for test
config.data.VoxCeleb1Verification.enroll_speaker_index = 6
config.data.VoxCeleb1Verification.eval_speaker_index = 7

config.data.ASVspoof2019 = edict()
config.data.ASVspoof2019.dataset = edict()
config.data.ASVspoof2019.dataloader = edict()
config.data.ASVspoof2019.dataset.data_file = '/home/wangli/ASGSR/dataset/ASVspoof2019_bonafide_speaker_veri.txt'
config.data.ASVspoof2019.dataset.train_path = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac'
config.data.ASVspoof2019.dataset.dev_path = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_dev/flac'
config.data.ASVspoof2019.dataset.eval_path = '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_eval/flac'
config.data.ASVspoof2019.dataloader.batch_size = 1
config.data.ASVspoof2019.enroll_waveform_index = 0
config.data.ASVspoof2019.eval_waveform_index = 1
config.data.ASVspoof2019.sample_rate_index = 2
config.data.ASVspoof2019.label_index = 3
config.data.ASVspoof2019.enroll_file_index = 4
config.data.ASVspoof2019.eval_file_index = 5  # spk1 for enroll, spk2 for test
config.data.ASVspoof2019.enroll_speaker_index = 6
config.data.ASVspoof2019.eval_speaker_index = 7

# ---------------------------------------- model ---------------------------------------- #
config.model = edict()
config.model.model_name = 'ResNetSE34V2'

# ResNetSE34V2
config.model.ResNetSE34V2 = edict()
config.model.ResNetSE34V2.save_path = '/home/wangli/ASGSR/pretrained_models/ResNetSE34V2/ResNetSE34V2.model'
config.model.ResNetSE34V2.config_path = '/home/wangli/ASGSR/voxceleb_trainer/configs/ResNetSE34V2.yaml'
config.model.ResNetSE34V2.threshold = 0.3347


# ECAPATDNN
config.model.ECAPATDNN = edict()
config.model.ECAPATDNN.save_path = '/mnt/data3/hehaorui/ckpt/ECAPATDNN/ECAPATDNN.pth'
config.model.ECAPATDNN.config_path = '/home/hehaorui/code/Amphion/ASGSR/voxceleb_trainer/configs/ECAPATDNN.yaml'
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

# ---------------------------------------- Attack ---------------------------------------
config.attack = edict()
config.attack.attack_name = 'PGD'

# FGSM
config.attack.FGSM = edict()
config.attack.FGSM.eps = 0.01
config.attack.FGSM.exp_mark = ['eps']
config.attack.FGSM.loss_function = edict()
config.attack.FGSM.loss_function.name = 'threshold_loss'
config.attack.FGSM.loss_function.threshold_loss = edict()
config.attack.FGSM.loss_function.threshold_loss.threshold = config.model[config.model.model_name].threshold
config.attack.FGSM.score_function = edict()
config.attack.FGSM.score_function.name = 'cosine_similarity_score'
config.attack.FGSM.decision_function = edict()
config.attack.FGSM.decision_function.name = 'cosine_similarity_decision'
config.attack.FGSM.decision_function.cosine_similarity_decision = edict()
config.attack.FGSM.decision_function.cosine_similarity_decision.threshold = config.model[config.model.model_name].threshold

# PGD
config.attack.PGD = edict()
config.attack.PGD.eps = 0.008
config.attack.PGD.alpha = 0.0004
config.attack.PGD.steps = 20
config.attack.PGD.random_start = True
config.attack.PGD.until_success = True
config.attack.PGD.exp_mark = ['eps', 'alpha', 'until_success' if config.attack.PGD.until_success else 'steps']
config.attack.PGD.loss_function = edict()
config.attack.PGD.loss_function.name = 'threshold_loss'
config.attack.PGD.loss_function.threshold_loss = edict()
config.attack.PGD.score_function = edict()
config.attack.PGD.score_function.name = 'cosine_similarity_score'
config.attack.PGD.score_function.cosine_similarity_score = edict()
config.attack.PGD.decision_function = edict()
config.attack.PGD.decision_function.name = 'cosine_similarity_decision'
config.attack.PGD.decision_function.cosine_similarity_decision = edict()

# FAKEBOB
config.attack.FAKEBOB = edict()
config.attack.FAKEBOB.threshold_estimated = None
config.attack.FAKEBOB.task = 'SV'
# config.attack.FAKEBOB.targeted = False
config.attack.FAKEBOB.confidence = 0
config.attack.FAKEBOB.epsilon = 0.005
config.attack.FAKEBOB.max_iter = 1000
config.attack.FAKEBOB.max_lr = 0.001
config.attack.FAKEBOB.min_lr = 1e-6
config.attack.FAKEBOB.samples_per_draw = 16
config.attack.FAKEBOB.samples_per_draw_batch_size = 16
config.attack.FAKEBOB.sigma = 0.001
config.attack.FAKEBOB.momentum = 0.9
config.attack.FAKEBOB.plateau_length = 5
config.attack.FAKEBOB.plateau_drop = 2.0
config.attack.FAKEBOB.stop_early = True
config.attack.FAKEBOB.stop_early_iter = 100
config.attack.FAKEBOB.batch_size = 1
config.attack.FAKEBOB.EOT_size = 1
config.attack.FAKEBOB.EOT_batch_size = 1
config.attack.FAKEBOB.thresh_est_wav_path = [
    ['/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac/PA_T_0005009.flac',
     '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac/PA_T_0004411.flac'],
    ['/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac/PA_T_0003405.flac',
     '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac/PA_T_0003825.flac'],
    ['/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac/PA_T_0000210.flac',
     '/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac/PA_T_0003636.flac']
]

config.attack.FAKEBOB.verbose = 1
config.attack.FAKEBOB.thresh_est_step = 0.1  # the smaller, the accurate, but the slower
