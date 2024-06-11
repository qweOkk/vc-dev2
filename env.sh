# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Install ffmpeg in Linux
conda install -y -c conda-forge ffmpeg

# Pip packages
pip install setuptools ruamel.yaml tqdm colorama easydict tabulate loguru json5 Cython unidecode inflect argparse g2p_en tgt librosa==0.9.1 matplotlib typeguard einops omegaconf hydra-core humanfriendly pandas

conda install pytorch==2.0.1 torchaudio==2.0.2 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y

pip install tensorboard tensorboardX accelerate==0.24.1 transformers diffusers praat-parselmouth audiomentations pedalboard ffmpeg-python==0.2.0 pyworld diffsptk nnAudio unidecode inflect ptwt

pip install torchmetrics pymcd openai-whisper frechet_audio_distance asteroid resemblyzer vector-quantize-pytorch==1.12.5

pip install https://github.com/vBaiCai/python-pesq/archive/master.zip

pip install fairseq

pip install git+https://github.com/lhotse-speech/lhotse

pip install -U encodec

pip install phonemizer==3.2.1 pypinyin==0.48.0

pip install black==24.1.1

# Uninstall nvidia-cublas-cu11 if there exist some bugs about CUDA version
#pip uninstall nvidia-cublas-cu11
#pip install nvidia-cuda-cupti-cu11==11.7.101 nvidia-cuda-nvrtc-cu11==11.7.99 nvidia-cuda-runtime-cu11==11.7.99 nvidia-cudnn-cu11==8.5.0.96 nvidia-cufft-cu11==10.9.0.58 nvidia-curand-cu11==10.2.10.91 nvidia-cusolver-cu11==11.4.0.1 nvidia-cusparse-cu11==11.7.4.91 nvidia-nccl-cu11==2.14.3 nvidia-nvtx-cu11==11.7.91                  