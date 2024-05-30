# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

 
export PYTHONPATH="./"
 


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
    exp_config="${exp_dir}"/exp_config_4gpu.json
fi


echo "Exprimental Configuration File: $exp_config"

# hubertnew="/mnt/petrelfs/hehaorui/data/ckpt/vc/newmhubert/model.safetensors"

hubertold="/mnt/data2/hehaorui/ckpt/zs-vc-ckpt/vc_mls_clean/model.safetensors"

whisperold="/mnt/data3/hehaorui/pretrained_models/VC/old_whisper/pytorch_model.bin"

hubert="/mnt/data2/hehaorui/ckpt/vc_new_exp/new_mhubert/checkpoint/epoch-0001_step-0496002_loss-0.567479/model.safetensors"
#hubert_se="/mnt/petrelfs/hehaorui/data/ckpt/vc/mhubert-noise-se/checkpoint/epoch-0000_step-0080000_loss-1.515860/pytorch_model.bin"
# whisper="/mnt/data2/hehaorui/ckpt/vc_new_exp/new_whisper/checkpoint/epoch-0000_step-0400001_loss-1.194134/model.safetensors"
# whisper_se="/mnt/data2/hehaorui/ckpt/vc_new_exp/new_whisper_aug/checkpoint/epoch-0000_step-0468003_loss-2.859798/model.safetensors"
# whisper_se_spk="/mnt/data2/hehaorui/ckpt/vc_new_exp/new_whisper_aug_spk/checkpoint/epoch-0000_step-0583003_loss-3.672843/model.safetensors"
hubert_se="/mnt/data2/hehaorui/ckpt/zs-vc-ckpt/epoch-0001_step-0796000_loss-0.567479/model.safetensors"
hubert_se_both="/mnt/data2/hehaorui/ckpt/vc_new_exp/new_mhubert_aug_spk_both/checkpoint/epoch-0001_step-0844000_loss-1.542532/model.safetensors"
cuda_id=7

checkpoint_path=$hubert_se

zero_shot_json_file_path="/mnt/data2/hehaorui/datasets/VCTK/zero_shot_json.json"
output_dir="/mnt/data2/hehaorui/exp_out"
vocoder_path="/mnt/data2/wangyuancheng/model_ckpts/ns2/bigvgan/g_00490000"
wavlm_path="/mnt/data3/hehaorui/pretrained_models/wavlm/wavlm-base-plus-sv"


echo "VC-Clean"
echo "CUDA ID: $cuda_id"
echo "Zero Shot Json File Path: $zero_shot_json_file_path"
echo "Checkpoint Path: $checkpoint_path"
echo "Output Directory: $output_dir"
echo "Vocoder Path: $vocoder_path"
echo "WavLM Path: $wavlm_path"

python "${work_dir}"/models/tts/vc/vc_inference.py \
    --config $exp_config \
    --checkpoint_path $checkpoint_path \
    --zero_shot_json_file_path $zero_shot_json_file_path \
    --output_dir $output_dir \
    --cuda_id ${cuda_id} \
    --vocoder_path $vocoder_path \
    --wavlm_path $wavlm_path \
    # --noisy
 
 
