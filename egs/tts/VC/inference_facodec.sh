
 

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

cuda_id=0
zero_shot_json_file_path="/mnt/data2/hehaorui/datasets/VCTK/zero_shot_json.json"
output_dir="/mnt/data2/hehaorui/exp_out"
vocoder_path="/mnt/data2/wangyuancheng/model_ckpts/ns2/bigvgan/g_00490000"
wavlm_path="/mnt/data3/hehaorui/pretrained_models/wavlm/wavlm-base-plus-sv"



echo "FACODEC"
echo "CUDA ID: $cuda_id"
echo "Zero Shot Json File Path: $zero_shot_json_file_path"
echo "Output Directory: $output_dir"
echo "WavLM Path: $wavlm_path"


python "${work_dir}"/models/tts/vc/facodec.py \
    --zero_shot_json_file_path $zero_shot_json_file_path \
    --output_dir $output_dir \
    --cuda_id ${cuda_id} \
    --wavlm_path $wavlm_path \
    --noisy
 
