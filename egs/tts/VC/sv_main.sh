# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


######## Build Experiment Environment ###########
exp_dir=$(cd `dirname $0`; pwd)
work_dir=$(dirname $(dirname $(dirname $exp_dir)))

export WORK_DIR=$work_dir
export PYTHONPATH=$work_dir
export PYTHONIOENCODING=UTF-8

if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}"/exp_config_2gpu.json
fi
echo "Exprimental Configuration File: $exp_config"

exp_name="trainSV"
cuda_id=6
######## Train Model ###########
echo "Exprimental Name: $exp_name"

#/mnt/data2/hehaorui/ckpt/zero-shot/epoch-0000_step-0100000_loss-0.078911/
#/mnt/data2/hehaorui/ckpt/zero-shot/epoch-0001_step-0400000_loss-0.037989/
#/mnt/data2/hehaorui/ckpt/vc/sv_se_vc/checkpoint/epoch-0001_step-0344000_loss-0.152027 #
#/mnt/data2/hehaorui/ckpt/vc/resume_vc_train/checkpoint/epoch-0002_step-0679000_loss-0.582957 # 最普通训练了最久
#/mnt/data2/hehaorui/ckpt/vc/train_speaker/checkpoint/epoch-0002_step-0509000_loss-2.241144//model.safetensors #有speaker CE loss
#/mnt/data2/hehaorui/ckpt/vc/sv_se_vc/checkpoint/epoch-0002_step-0749000_loss-0.136304/model.safetensors #有contrastive loss

checkpoint_path="/mnt/data2/hehaorui/ckpt/zero-shot/epoch-0001_step-0400000_loss-0.037989/model.safetensors"
python "${work_dir}"/models/sv/sv_trainer.py \
    --config $exp_config \
    --checkpoint_path $checkpoint_path \
    --cuda_id ${cuda_id} 
