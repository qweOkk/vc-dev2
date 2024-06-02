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
# 配置要改
if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}"/exp_config_4gpu_both_noise.json
fi
echo "Exprimental Configuration File: $exp_config"

#这里需要跟exp_config保持对应
exp_name="new_mhubert_both_noise"

#训练的时候用0，1，2，3
#测试的时候用6/7
if [ -z "$gpu" ]; then
    gpu="0,1,2,3"
fi

######## Train Model ###########
echo "Exprimental Name: $exp_name"

# 端口号每次训练都要改 26666 26667 26668 26669
# --mixed_precision fp16

# # 这是从头开始训练
# CUDA_VISIBLE_DEVICES=$gpu accelerate launch --main_process_port 26666 --mixed_precision fp16 \
# "${work_dir}"/bins/tts/train.py \
#     --config $exp_config \
#     --exp_name $exp_name \
#     --log_level debug 

# # 这是resume训练
# CUDA_VISIBLE_DEVICES=$gpu accelerate launch --main_process_port 26666 --mixed_precision fp16 \
# "${work_dir}"/bins/tts/train.py \
#     --config $exp_config \
#     --exp_name $exp_name \
#     --log_level debug \
#     --resume \
#     --resume_type resume \

# 这是指定一个checkpoint 文件夹
CUDA_VISIBLE_DEVICES=$gpu accelerate launch --main_process_port 26666 --mixed_precision fp16 \
"${work_dir}"/bins/tts/train.py \
    --config $exp_config \
    --exp_name $exp_name \
    --log_level debug \
    --resume \
    --resume_type resume \
    --checkpoint_path /mnt/data2/hehaorui/ckpt/vc_new_exp/new_mhubert/checkpoint/epoch-0001_step-0496002_loss-0.567479




 