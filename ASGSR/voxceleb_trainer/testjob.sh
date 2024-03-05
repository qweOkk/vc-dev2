#!/bin/bash
#SBATCH -J testASV              # 作业名是 train
#SBATCH -p p-RTX2080          # 提交到 默认的defq 队列
#SBATCH -N 1                 # 使用2个节点
#SBATCH --ntasks-per-node=4 # 每个节点开启8个进程
#SBATCH --cpus-per-task=1    # 每个进程占用一个 cpu 核心
#SBATCH -t 4800:00             # 任务最大运行时间是 50 分钟
#SBATCH --gres=gpu:1         # 如果是gpu任务需要在此行定义gpu数量,此处为1

#python ./trainSpeakerNet.py --eval --model ResNetSE34V2 --log_input True --encoder_type ASP --n_mels 64 --trainfunc softmaxproto --save_path exps/test --eval_frames 400  --initial_model /home/wangli/ASGSR/pretrained_models/ResNetSE34V2/ResNetSE34V2.model
#python ./trainSpeakerNet.py --eval --config ./configs/RawNet3_AAM.yaml --initial_model models/weights/RawNet3/model.pt
#python ./trainSpeakerNet.py --eval --config ./configs/ECAPATDNN.yaml --initial_model /home/wangli/ASGSR/pretrained_models/ECAPATDNN/ECAPATDNN.pth
#python ./trainSpeakerNet.py --eval --config ./configs/XVector_AAM.yaml --initial_model /home/wangli/ASGSR/voxceleb_trainer/exps/XVector_AAM/model/model000000230.model
python ./trainSpeakerNet.py --eval --config ./configs/ECAPATDNN_AAM.yaml --initial_model /home/wangli/ASGSR/voxceleb_trainer/exps/ECAPATDNN_AAM/model/model000000060.model