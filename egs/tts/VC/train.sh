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

if [ -z "$exp_config" ]; then
    exp_config="${exp_dir}"/exp_config_4gpu_clean.json
fi
echo "Exprimental Configuration File: $exp_config"

exp_name="new_mhubert"

if [ -z "$gpu" ]; then
    gpu="0,1,2,3"
fi

######## Train Model ###########
echo "Exprimental Name: $exp_name"

CUDA_VISIBLE_DEVICES=$gpu accelerate launch --main_process_port 28500 \
"${work_dir}"/bins/tts/train.py \
    --config $exp_config \
    --exp_name $exp_name \
    --log_level debug \
    --resume \
    --resume_type resume \
    --checkpoint_path /nfsmnt/qiujunwen/ckpt/vc_new_exp/new_mhubert/checkpoint/final_epoch-0400_step-0002800_loss-41.564853

    # --resume \
    # --resume_type resume \
    