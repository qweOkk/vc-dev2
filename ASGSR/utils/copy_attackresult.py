import os
import shutil

# ****** only copy target attack **********

# copy attack result files to the record dir
src_dirs = [
    '/home/wangli/ASGSR/attack/attack_exps/VoxCeleb1Verification/PGD_ECAPATDNN-ECAPATDNN_eps-0.008_alpha-0.0004_steps-20',
    '/home/wangli/ASGSR/attack/attack_exps/VoxCeleb1Verification/PGD_RawNet3-model_eps-0.008_alpha-0.0004_steps-20',
    '/home/wangli/ASGSR/attack/attack_exps/VoxCeleb1Verification/PGD_ResNetSE34V2-ResNetSE34V2_eps-0.008_alpha-0.0004_steps-20',
    '/home/wangli/ASGSR/attack/attack_exps/VoxCeleb1Verification/PGD_XVector-model000000230_eps-0.008_alpha-0.0004_steps-20',
    '/home/wangli/ASGSR/attack/ensemble_attack_exps/VoxCeleb1Verification/PGD_ResNetSE34V2-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727170536',
    '/home/wangli/ASGSR/attack/ensemble_attack_exps/VoxCeleb1Verification/PGD_XVector-PGD_ECAPATDNN-PGD_RawNet3-Step20-20230727171124',
    '/home/wangli/ASGSR/attack/ensemble_attack_exps/VoxCeleb1Verification/PGD_XVector-PGD_ResNetSE34V2-PGD_ECAPATDNN-Step20-20230727171240',
    '/home/wangli/ASGSR/attack/ensemble_attack_exps/VoxCeleb1Verification/PGD_XVector-PGD_ResNetSE34V2-PGD_RawNet3-Step20-20230727171220'
]

dst_dirs = [
    '/mntcephfs/lab_data/wangli/ASGSR/record/VoxCeleb1Verification'
]

for src_dir in src_dirs:
    s = src_dir.split('/')[-1]
    for dst_dir in dst_dirs:
        for root, dirs, files in os.walk(dst_dir):
            for dir in dirs:
                if s in dir:
                    # os.system('rm {}'.format(os.path.join(root, dir, 'attackResult.txt')))
                    res = []
                    with open(os.path.join(src_dir, 'attackResult.txt'), 'r') as f:
                        for line in f.readlines():
                            if line.split()[3] == '0': # only copy target attack
                                res.append(line)
                    with open(os.path.join(root, dir, 'attackResult.txt'), 'w') as f:
                        f.writelines(res)
                    print('write to {}'.format(os.path.join(root, dir, 'attackResult.txt')))
                    break
