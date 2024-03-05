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
    exp_config="${exp_dir}"/exp_config_6gpu.json
fi
echo "Exprimental Configuration File: $exp_config"

exp_name="sv_VCTK_voxceleb_voxcelebclean"
cuda_id=7
######## Train Model ###########
echo "Exprimental Name: $exp_name"

python "${work_dir}"/models/tts/vc/sv_inference.py \
    --config $exp_config \
    --exp_name $exp_name \
    --checkpoint_path_1 "/mnt/data2/hehaorui/ckpt/zero-shot/epoch-0001_step-0400000_loss-0.037989/model.safetensors" \
    --checkpoint_path_2 "/mnt/data2/hehaorui/ckpt/vc/resume_vc_train/checkpoint/epoch-0002_step-0679000_loss-0.582957/model.safetensors" \
    --cuda_id ${cuda_id} \
    --test_set "voxceleb" 

python "${work_dir}"/models/tts/vc/sv_inference.py \
    --config $exp_config \
    --exp_name $exp_name \
    --checkpoint_path_1 "/mnt/data2/hehaorui/ckpt/zero-shot/epoch-0001_step-0400000_loss-0.037989/model.safetensors" \
    --checkpoint_path_2 "/mnt/data2/hehaorui/ckpt/vc/resume_vc_train/checkpoint/epoch-0002_step-0679000_loss-0.582957/model.safetensors" \
    --cuda_id ${cuda_id} \
    --test_set "voxceleb_clean" 

python "${work_dir}"/models/tts/vc/sv_inference.py \
    --config $exp_config \
    --exp_name $exp_name \
    --checkpoint_path_1 "/mnt/data2/hehaorui/ckpt/zero-shot/epoch-0001_step-0400000_loss-0.037989/model.safetensors" \
    --checkpoint_path_2 "/mnt/data2/hehaorui/ckpt/vc/resume_vc_train/checkpoint/epoch-0002_step-0679000_loss-0.582957/model.safetensors" \
    --cuda_id ${cuda_id} \
    --test_set "VCTK" 



#/mnt/data2/hehaorui/ckpt/zero-shot/epoch-0000_step-0100000_loss-0.078911/
#/mnt/data2/hehaorui/ckpt/zero-shot/epoch-0001_step-0400000_loss-0.037989/
#/mnt/data2/hehaorui/ckpt/vc/sv_se_vc/checkpoint/epoch-0001_step-0344000_loss-0.152027
#/mnt/data2/hehaorui/ckpt/vc/resume_vc_train/checkpoint/epoch-0002_step-0587000_loss-0.565451
#/mnt/data2/hehaorui/ckpt/vc/resume_vc_train/checkpoint/epoch-0002_step-0679000_loss-0.582957