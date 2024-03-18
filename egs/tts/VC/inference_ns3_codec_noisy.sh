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

exp_name="ns3_codec_vc_inference"

######## Train Model ###########
echo "Exprimental Name: $exp_name"

python "${work_dir}"/models/tts/vc/facodec_vc_inference_noisy.py\
    --output_dir "/mnt/data2/hehaorui/vc_test/Results/NS3_codec_noisy" \
    --cuda_id 7 

#/mnt/data2/hehaorui/ckpt/zero-shot/epoch-0000_step-0100000_loss-0.078911/
#/mnt/data2/hehaorui/ckpt/zero-shot/epoch-0001_step-0400000_loss-0.037989/
#/mnt/data2/hehaorui/ckpt/vc/sv_se_vc/checkpoint/epoch-0001_step-0344000_loss-0.152027