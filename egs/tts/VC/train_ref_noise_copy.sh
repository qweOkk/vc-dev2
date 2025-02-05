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
 
cd $work_dir/modules/monotonic_align
mkdir -p monotonic_align
python setup.py build_ext --inplace
cd $work_dir

# 这一行开始
# 3 clean，reference noisy, both noisy
if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}"/exp_config_4gpu_both_noise.json
fi
echo "Exprimental Configuration File: $exp_config"

#这里需要跟exp_config保持对应
exp_name="new_mhubert_ref_noise"
# exp_name="new_mhubert_ref_noisy"

#训练的时候用0，1，2，3
#测试的时候用6/7
if [ -z "$gpu" ]; then
    gpu="6,7"
fi

######## Train Model ###########
echo "Exprimental Name: $exp_name"

# 端口号每次训练都要改 26666 26667 26668 26669
# --mixed_precision fp16

# # 这是从头开始训练实验4/5是这个
# CUDA_VISIBLE_DEVICES=$gpu accelerate launch --main_process_port 26666 --mixed_precision fp16 \
# "${work_dir}"/bins/tts/train.py \
#     --config $exp_config \
#     --exp_name $exp_name \
#     --log_level debug 

#虚拟环境conda activate vc

# # 这是resume训练, 实验1是这个
# CUDA_VISIBLE_DEVICES=$gpu accelerate launch --main_process_port 26666 --mixed_precision fp16 \
# "${work_dir}"/bins/tts/train.py \
#     --config $exp_config \
#     --exp_nam $exp_name \
#     --log_level debug \
#     --resume \
#     --resume_type resume \

# 这是指定一个checkpoint 文件夹，实验2/3是这个
CUDA_VISIBLE_DEVICES=$gpu accelerate launch --main_process_port 26667 --mixed_precision fp16 \
"${work_dir}"/bins/tts/train.py \
    --config $exp_config \
    --exp_name $exp_name \
    --log_level debug \
    --resume \
    --resume_type resume \
    --checkpoint_path /mnt/data2/hehaorui/ckpt/vc_new_exp/new_mhubert/checkpoint/epoch-0001_step-0496002_loss-0.567479