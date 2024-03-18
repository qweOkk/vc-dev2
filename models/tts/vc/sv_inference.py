import argparse
import torch
import numpy as np
import torch
from tqdm import tqdm
from safetensors.torch import load_model
from utils.util import load_config
import os
from models.tts.vc.vc_trainer import VCTrainer, mel_spectrogram
from utils.util import load_config
from models.tts.vc.ns2_uniamphion import UniAmphionVC
from models.tts.vc.sv_scirpts.sv_utils import metric, get_sv_data, get_batch_cos_sim
from ASGSR.attack.attackMain import get_baseline_model
import librosa
from denoiser import pretrained
import warnings
warnings.filterwarnings('ignore')

def build_trainer(args, cfg):
    supported_trainer = {
        "VC": VCTrainer,
    }
    trainer_class = supported_trainer[cfg.model_type]
    trainer = trainer_class(args, cfg)
    return trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config.json",
        help="json files for configurations.",
        required=True,
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of dataloader workers."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="exp_name",
        help="A specific name to note the experiment",
        required=True,
    )
    parser.add_argument(
        "--checkpoint_path_1",
        type=str,
        required=True,
        help="Checkpoint for resume training or finetuning.",
    )
    parser.add_argument(
        "--checkpoint_path_2",
        type=str,
        required=True,
        help="Checkpoint for resume training or finetuning.",
    )
    parser.add_argument(
        "--test_set",
        type=str,
        default="VCTK",
        help="SV test set",
    )
    parser.add_argument(
        "--cuda_id", 
        type=int, 
        default=7, 
        help="Cuda id for training."
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()

    if args.test_set == "VCTK":
        text_path = '/mnt/data2/hehaorui/vctk_sv_test.txt'
        folder_path = '/mnt/data2/hehaorui/VCTK/wav48'
        split_rate = 0.05
    elif args.test_set == "voxceleb":
        text_path = '/mnt/data2/hehaorui/VoxCeleb1/veri_test.txt'
        folder_path = '/mnt/data2/hehaorui/VoxCeleb1/wav/'
        split_rate = 0.25
    elif args.test_set == "voxceleb_clean":
        text_path = '/mnt/data2/hehaorui/VoxCeleb1/veri_test_clean.txt'
        folder_path = '/mnt/data2/hehaorui/VoxCeleb1/wav/'
        split_rate =  0.25
    cfg = load_config(args.config)
    cfg.exp_name = args.exp_name
    
    cuda_id = args.cuda_id
    args.local_rank = torch.device(f"cuda:{cuda_id}")
    print("local rank", args.local_rank)
    ckpt_path = args.checkpoint_path_1
    print("ckpt_path", ckpt_path)
    # embty the chosen device memory
    with torch.cuda.device(args.local_rank):
        torch.cuda.empty_cache()
    model = UniAmphionVC(cfg.model)
    print("loading model")
    load_model(model, ckpt_path)
    print("model loaded")
    model.cuda(args.local_rank)
    model.eval()

    model_se = UniAmphionVC(cfg=cfg.model, use_speaker = True, speaker_num = 12954)
    ckpt_path = args.checkpoint_path_2 # contrastive learning model
    print("ckpt_path", ckpt_path)
    print("loading model")
    load_model(model_se, ckpt_path)
    print("model loaded")
    model_se.cuda(args.local_rank)
    model_se.eval()

    '''prepare baseline'''
    baseline_model = get_baseline_model("ECAPATDNN")
    baseline_model.cuda(args.local_rank)
    baseline_model.eval()

    data = get_sv_data(text_path) #从txt文件中读出df
    data = data.sample(frac=split_rate, random_state=1126).reset_index(drop=True) #shuffle df
    print(len(data))
    
    labels_all = []
    socres_bsl_all = []
    scores_sv_all = []
    scores_all = []

    # denoiser
    denoiser_model = pretrained.dns64().cuda(args.local_rank)


    print("--- start testing ---")
    for idx in tqdm(range(len(data))):
        first_wav = data['First'].iloc[idx]
        second_wav = data['Second'].iloc[idx]
        if not os.path.isabs(first_wav):
            first_wav = os.path.join(folder_path, first_wav)
        if not os.path.isabs(second_wav):
            second_wav = os.path.join(folder_path, second_wav)
        
        label = torch.tensor(int(data['Label'].iloc[idx]))
        with torch.no_grad():
            first_wav, _ = librosa.load(first_wav, sr=16000)
            second_wav, _ = librosa.load(second_wav, sr=16000)

            # 取前5s
            first_wav = first_wav[:16000*5]
            second_wav = second_wav[:16000*5]


            # first_wav = torch.from_numpy(first_wav).to(args.local_rank).unsqueeze(0)
            # second_wav = torch.from_numpy(second_wav).to(args.local_rank).unsqueeze(0)
            # first_wav = convert_audio(first_wav, 16000, denoiser_model.sample_rate, denoiser_model.chin)
            # second_wav = convert_audio(second_wav, 16000, denoiser_model.sample_rate, denoiser_model.chin)

            # with torch.no_grad():
            #     first_wav = denoiser_model(first_wav[None])[0]
            #     second_wav = denoiser_model(second_wav[None])[0]
            # #resample to 16000
            # first_wav = torchaudio.transforms.Resample(denoiser_model.sample_rate, 16000)(first_wav)
            # second_wav = torchaudio.transforms.Resample(denoiser_model.sample_rate, 16000)(second_wav)
            # first_wav = first_wav.cpu().numpy().squeeze()
            # #print(first_wav.shape)
            # second_wav = second_wav.cpu().numpy().squeeze()

            first_wav = np.pad(first_wav, (0, 1600 - len(first_wav) % 1600))
            first_audio = torch.from_numpy(first_wav).to(args.local_rank)
            first_audio = first_audio[None, :]
            # Load and process audio data
            first_mel = mel_spectrogram(first_audio, n_fft=1024,
                    num_mels=80,
                    sampling_rate=16000,
                    hop_size=200,
                    win_size=800,
                    fmin=0,
                    fmax=8000,
                    )
            first_mel = first_mel.transpose(1, 2).to(device=args.local_rank)
            first_mask = torch.ones(first_mel.shape[0], first_mel.shape[1]).to(args.local_rank)

            second_wav = np.pad(second_wav, (0, 1600 - len(second_wav) % 1600))
            second_audio = torch.from_numpy(second_wav).to(args.local_rank)
            second_audio = second_audio[None, :]
            # Load and process audio data
            second_mel = mel_spectrogram(second_audio, n_fft=1024,
                    num_mels=80,
                    sampling_rate=16000,
                    hop_size=200,
                    win_size=800,
                    fmin=0,
                    fmax=8000,
                    )
            second_mel = second_mel.transpose(1, 2).to(device=args.local_rank)
            second_mask = torch.ones(second_mel.shape[0], second_mel.shape[1]).to(args.local_rank)

            first_emb = model.sv_inference(first_mel, first_mask)
            #first_emb = torch.mean(first_emb, dim=1) # (B, 512) emb for the fist wav
            second_emb = model.sv_inference(second_mel, second_mask)
            #second_emb = torch.mean(second_emb, dim=1) # (B, 512) emb for the second wav
            score = get_batch_cos_sim(first_emb,second_emb).to(torch.device('cpu'))

            first_emb_sv = model_se.sv_inference(first_mel, first_mask)
            #first_emb_sv = torch.mean(first_emb_sv, dim=1) # (B, 512) emb for the fist wav
            second_emb_sv = model_se.sv_inference(second_mel, second_mask)
            #second_emb_sv = torch.mean(second_emb_sv, dim=1) # (B, 512) emb for the second wav
            score_sv = get_batch_cos_sim(first_emb_sv,second_emb_sv).to(torch.device('cpu'))

            first_emb_bsl = baseline_model(first_audio.to(device=args.local_rank))
            second_emb_bsl = baseline_model(second_audio.to(device=args.local_rank))
            scores_bsl = get_batch_cos_sim(first_emb_bsl,second_emb_bsl).to(torch.device('cpu'))
        
            scores_all.append(score)
            scores_sv_all.append(score_sv)
            socres_bsl_all.append(scores_bsl)
            labels_all.append(label)
    print("--- testing finished ---")
    
    scores_all = np.array(scores_all)
    scores_all = scores_all.reshape(-1)
    scores_sv_all = np.array(scores_sv_all)
    scores_sv_all = scores_sv_all.reshape(-1)
    socres_bsl_all = np.array(socres_bsl_all)
    socres_bsl_all = socres_bsl_all.reshape(-1)
    labels_all = np.array(labels_all)

    print("---- ECAPATDNN----")
    metric(socres_bsl_all, labels_all)

    print("---- Our Baseline ----")
    metric(scores_all,labels_all)

    print("---- Our Baseline + Loss ----")
    metric(scores_sv_all,labels_all)
        
        

   
if __name__ == "__main__":
    main()