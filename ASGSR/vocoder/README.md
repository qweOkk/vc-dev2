# Vocoders System

We use 6 types of vocoder for audio synthesis, namely world, HiFiGAN, WaveRNN, DiffWave, WaveNet, WaneGlow.

The official code implementations
of [world](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder), [WaveNet](https://github.com/r9y9/wavenet_vocoder)
and [WaneGlow](https://github.com/NVIDIA/waveglow) are used respectively.

HiFiGAN, WaveRNN and DiffWave are implemented using an internal framework (not yet open source).

## 环境

environment.yaml

requirements.txt

## Dataset

Train: ASVspoof2019 PA train set, bonafide audio.
Inference: ASVspoof2019 PA, all bonafide audio.

### ffmpeg 转 wav

```
同目录层级转换，比如ASVspoof2019_PA_dev/flac 转换到 ASVspoof2019_PA_dev/wav
ffmpeg -i inputfile.flac output.wav
```

## HiFiGAN, WaveRNN and DiffWave

这三个vocoder都在Singing-Voice-Conversion-System这个目录中实现

### Preprocess

修改egs/nonparallel/config.yaml中output_path为合适的路径，然后运行

   ```bash
   python egs/nonparallel/preprocess.py --yaml "egs/nonparallel/config.yaml"
   ```

上述代码将预处理ASVspoof2019然后提取训练vocoder所用特征

### Training Vocoder

   ```bash
   python egs/nonparallel/train_vocoder.py --yaml "egs/nonparallel/config.yaml" --vocoder_yaml "[vocoder configuration file]"
   ```

    vocoder_yaml参数指定为对应的vocoder配置文件，如chkszsvc/config/vocoder/config/nsfhifigan.yaml

### Inference (Conversion)

   ```bash
   python egs/nonparallel/inference_vocoder_e2e.py --checkpoint_of_vocoder "[cheakpoint folder]"

   ```
   checkpoint_of_vocoder参数指定为训练好的vocoder模型路径，如"/mntnfs/lee_data1/wangli/ASGSR/ASVspoof2019/vocoder/model_vocoder/ckpts/ASVspoof2019/mels_16000hz-mels_wavernn_general_pretrained:False_lr_0.0002_dspmatch_True"

## WaveGlow

### datalist 准备

下载完ASVspoof2019数据后，按照ASVspoof2019的路径修改waveglow目录下的ASVspoof2019.py文件中的train_file train_path dev_file
dev_path test_file test_path，然后运行ASVspoof2019.py文件

生成训练所需的 train_files.txt 和 test_files.txt

### 特征提取

python mel2samp.py -f test_files.txt -o outputdir -c config.json

-o 制定特征输出文件路径

ls outputdir/*.pt > mel_files.txt

### 训练

python train.py -c config.json

### 生成audio

python3 inference.py -f mel_files.txt -w /mntnfs/lee_data1/wangli/ASGSR/ASVspoof2019/waveglow/waveglow_62000 -o
inference_wav --is_fp16 -s 0.6 --sampling_rate 16000 -d 0.1

-w 是指定训练完成的模型路径

-o指定输出audio路径

## WaveNet

### 划分训练、验证、测试集

WaveNet需要将训练，验证以及测试集分别划分到3个文件夹中，分别为train_no_dev，dev，test，ASVspoof2019.py实现此功能

修改ASVspoof2019.py中的train_file、train_path以及train_dst_path

train_file指的是官方的训练集文件ASVspoof2019.PA.cm.train.trn.txt的存储路径

train_path是文件存储路径，如/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/wav

train_dst_path是用于wavenet训练的文件存储路径，后几级的文件必须为data/asvspoof2019/train_no_dev，如/root/data/asvspoof2019/train_no_dev

### 修改配置文件egs/mulaw256/run.sh

修改WaveNet声码器文件夹中的egs/mulaw256/run.sh文件

dumpdir为保存中间文件，自行指定即可

data_root为上一步的路径，如data/

eval_checkpoint为模型的路径，指定为训练好的模型即可

### 提取特征

./run.sh --stage 1 --stop-stage 1

### 训练

./run.sh --stage 2 --stop-stage 2

训练中可以增加--checkpoint参数，继续训练之前训练的模型

### 生成音频

./run.sh --stage 3 --stop-stage 3

## World

Based on the code.