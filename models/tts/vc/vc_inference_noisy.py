import argparse
import torch
import numpy as np
import torch
import random
from tqdm import tqdm
from safetensors.torch import load_model
import librosa
from utils.util import load_config
import os
import json
from models.tts.vc.vc_trainer import VCTrainer, mel_spectrogram, extract_world_f0
from utils.util import load_config
from models.tts.vc.ns2_uniamphion import UniAmphionVC
from models.tts.vc.hubert_kmeans import HubertWithKmeans


import torch
from denoiser import pretrained
from denoiser.dsp import convert_audio


semodel = pretrained.dns64().cuda()

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
        "--resume",
        type=str,
        default=None,
        # action="store_true",
        help="The model name to restore",
    )
    parser.add_argument(
        "--log_level", default="info", help="logging level (info, debug, warning)"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Checkpoint for resume training or finetuning.",
    )
    parser.add_argument(
        "--output_dir", 
        help="output path",
        required=True,
    )
    parser.add_argument(
        "--zero_shot_json_file_path",
        type=str,
        default="/home/hehaorui/code/Amphion/egs/tts/VC/zero_shot_json.json",
        help="Zero shot json file path",
    )
    parser.add_argument(
        "--cuda_id", 
        type=int, 
        default=7, 
        help="Cuda id for training."
    )

    parser.add_argument("--stdout_interval", default=5, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.exp_name = args.exp_name
    random.seed(20011126)
    print("loading config")
    
    test_noise_dir = "/home/hehaorui/code/Amphion/MS-SNSD/noise_test"

    def get_noisy_wavforms(directory):
        flac_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                # flac or wav
                if file.endswith(".flac") or file.endswith(".wav"):
                    flac_files.append(os.path.join(root, file))
        return flac_files

    noise_filenames = get_noisy_wavforms(test_noise_dir)
    print("noise_filenames len", len(noise_filenames))
    # generate SNR from 1.0, 1.5 to 5.0
    SNR = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    print("SNR", SNR)

    def snr_mixer(clean, noise, snr):
            # Normalizing to -25 dB FS
            rmsclean = (clean**2).mean()**0.5
            epsilon = 1e-10
            rmsclean = max(rmsclean, epsilon)
            scalarclean = 10 ** (-25 / 20) / rmsclean
            clean = clean * scalarclean

            rmsnoise = (noise**2).mean()**0.5
            scalarnoise = 10 ** (-25 / 20) /rmsnoise
            noise = noise * scalarnoise
            rmsnoise = (noise**2).mean()**0.5
            
            # Set the noise level for a given SNR
            noisescalar = np.sqrt(rmsclean / (10**(snr/20)) / rmsnoise)
            noisenewlevel = noise * noisescalar
            noisyspeech = clean + noisenewlevel
            noisyspeech_tensor = torch.tensor(noisyspeech, dtype=torch.float32)
            return noisyspeech_tensor

    def add_noise(clean):
        # self.noise_filenames: list of noise files
        # self.SNR: list of SNR = np.linspace(int(snr_lower), int(snr_upper), int(total_snrlevels))
        random_idx = np.random.randint(0, np.size(noise_filenames))
        noise, _ = librosa.load(noise_filenames[random_idx], sr=16000)
        if len(noise)>=len(clean):
            noise = noise[0:len(clean)] #截取噪声的长度
        else:
            while len(noise)<=len(clean): #如果噪声的长度小于语音的长度
                random_idx = (random_idx + 1)%len(noise_filenames) #随机读一个噪声
                newnoise, fs = librosa.load(noise_filenames[random_idx], sr=16000)
                noiseconcat = np.append(noise, np.zeros(int(fs * 0.2)))#在噪声后面加上0.2静音
                noise = np.append(noiseconcat, newnoise)#拼接噪声
        noise = noise[0:len(clean)] #截取噪声的长度
        random_SNR_idx = np.random.randint(0, np.size(SNR)) #随机选择一个SNR
        noisyspeech = snr_mixer(clean=clean, noise=noise, snr=SNR[random_SNR_idx]) #根据随机的SNR级别，混合生成带噪音频
        del noise
        return noisyspeech

    # Model saving dir
    args.log_dir = os.path.join(cfg.log_dir, args.exp_name)
    os.makedirs(args.log_dir, exist_ok=True)

    cuda_id = args.cuda_id
    args.local_rank = torch.device(f"cuda:{cuda_id}")
    print("local rank", args.local_rank)
    ckpt_path = args.checkpoint_path
    print("ckpt_path", ckpt_path)
    zero_shot_json_file_path = args.zero_shot_json_file_path
    print("zero_shot_json_file_path", zero_shot_json_file_path)
    with torch.cuda.device(args.local_rank):
        torch.cuda.empty_cache()
    model = UniAmphionVC(cfg.model)
    print("loading model")
    load_model(model, ckpt_path)
    print("model loaded")

    print("loading enhancement_model")
    enhancement_model = pretrained.dns64().cuda(args.local_rank)
    print("enhancement_model loaded")

    model.cuda(args.local_rank)
    model.eval()
    
    w2v = HubertWithKmeans(
            checkpoint_path="/mnt/data3/hehaorui/ckpt/mhubert/mhubert_base_vp_en_es_fr_it3.pt",
            kmeans_path="/mnt/data3/hehaorui/ckpt/mhubert/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin",
        )
    w2v = w2v.to(device=args.local_rank)
    w2v.eval()

    print("loading zero shot json")
    with open(zero_shot_json_file_path, "r") as f:
        zero_shot_json = json.load(f)
    zero_shot_json = zero_shot_json["test_cases"]
    print("length of test cases", len(zero_shot_json))

    utt_dict = {}
    for info in zero_shot_json:
        utt_id = info["uid"]
        utt_dict[utt_id] = {}
        utt_dict[utt_id]["source_speech"] = info["source_wav_path"]
        utt_dict[utt_id]["target_speech"] = info["target_wav_path"]
        utt_dict[utt_id]["prompt_speech"] = info["prompt_wav_path"]

    os.makedirs(args.output_dir, exist_ok=True)
    test_cases = []

    os.makedirs(f"{args.output_dir}/recon/mel", exist_ok=True)
    os.makedirs(f"{args.output_dir}/target/mel", exist_ok=True)
    os.makedirs(f"{args.output_dir}/source/mel", exist_ok=True)
    os.makedirs(f"{args.output_dir}/prompt/mel", exist_ok=True)

    temp_id = 0
    for utt_id, utt in tqdm(utt_dict.items()):
        # if temp_id > 10:
        #     break
        temp_id += 1
        # source is the input
        wav_path = utt["source_speech"]
        wav, _ = librosa.load(wav_path, sr=16000)
        wav = np.pad(wav, (0, 1600 - len(wav) % 1600))
        audio = torch.from_numpy(wav).to(args.local_rank)
        audio = audio[None, :]
        
        # target is the ground truth
        tgt_wav_path = utt["target_speech"]
        tgt_wav,_ = librosa.load(tgt_wav_path, sr=16000)
        tgt_wav = np.pad(tgt_wav, (0, 1600 - len(tgt_wav) % 1600))
        tgt_audio = torch.from_numpy(tgt_wav).to(args.local_rank)
        tgt_audio = tgt_audio[None, :]

        # prompt is the reference
        ref_wav_path = utt["prompt_speech"]
        ref_wav,_ = librosa.load(ref_wav_path, sr=16000)
        # 给prompt加噪音————————————————————！！！！
        ref_wav = add_noise(ref_wav)
        ref_wav = np.pad(ref_wav, (0, 200 - len(ref_wav) % 200))
        ref_audio = torch.from_numpy(ref_wav).to(args.local_rank)

        # original_shape = ref_audio[None, :].shape
        # #再用enhancement模型去噪
        # ref_audio = ref_audio.unsqueeze(0)
        # ref_audio = convert_audio(ref_audio, 16000, enhancement_model.sample_rate, enhancement_model.chin)
        # with torch.no_grad():
        #     denoised_ref = enhancement_model(ref_audio[None])[0] #1
        
        # #判断是否存在denoised_ref这个变量
        # if "denoised_ref" in locals():
        #     ref_audio = denoised_ref
        # else:
        #     ref_audio = ref_audio[None, :]
        
        ref_audio = ref_audio[None, :]
        with torch.no_grad():
            ref_mel = mel_spectrogram(
                ref_audio,
                n_fft=1024,
                num_mels=80,
                sampling_rate=16000,
                hop_size=200,
                win_size=800,
                fmin=0,
                fmax=8000,
            )
            tgt_mel = mel_spectrogram(
                tgt_audio,
                n_fft=1024,
                num_mels=80,
                sampling_rate=16000,
                hop_size=200,
                win_size=800,
                fmin=0,
                fmax=8000,
            )
            source_mel = mel_spectrogram(
                audio,
                n_fft=1024,
                num_mels=80,
                sampling_rate=16000,
                hop_size=200,
                win_size=800,
                fmin=0,
                fmax=8000,
            )
            ref_mel = ref_mel.transpose(1, 2).to(device=args.local_rank)
     
            ref_mask = torch.ones(ref_mel.shape[0], ref_mel.shape[1]).to(args.local_rank)

            _, content_feature = w2v(audio)
            content_feature = content_feature.to(device=args.local_rank)

            pitch = extract_world_f0(audio).to(device=args.local_rank)
            pitch = (pitch - pitch.mean(dim=1, keepdim=True)) / (
                pitch.std(dim=1, keepdim=True) + 1e-6
            )

            x0 = model.inference(
                content_feature=content_feature,
                pitch=pitch,
                x_ref=ref_mel,
                x_ref_mask=ref_mask,
                inference_steps=200,
                sigma=1.2,
            )

            test_case = dict()
            recon_path = f"{args.output_dir}/recon/mel/recon_{utt_id}.npy"
            ref_path = f"{args.output_dir}/target/mel/target_{utt_id}.npy"
            source_path = f"{args.output_dir}/source/mel/source_{utt_id}.npy"
            prompt_path = f"{args.output_dir}/prompt/mel/prompt_{utt_id}.npy"
            test_case["recon_ref_wav_path"] = recon_path.replace("/mel/", "/wav/").replace(".npy", "_generated_e2e.wav")
            test_case["reference_wav_path"] = ref_path.replace("/mel/", "/wav/").replace(".npy", "_generated_e2e.wav")
            np.save(recon_path, x0.transpose(1, 2).detach().cpu().numpy())
            np.save(prompt_path, ref_mel.transpose(1, 2).detach().cpu().numpy())
            np.save(ref_path, tgt_mel.detach().cpu().numpy())
            np.save(source_path, source_mel.detach().cpu().numpy())
            test_cases.append(test_case)
    del model, w2v, ref_mel, ref_mask, content_feature, pitch, x0, ref_audio, tgt_audio, audio, tgt_mel, source_mel
    
    data = dict()
    data["dataset"] = "recon"
    data["test_cases"] = test_cases
    with open(f"{args.output_dir}/recon.json", "w") as f:
        json.dump(data, f)
    with torch.device(f"cuda:{cuda_id}"):
        torch.cuda.empty_cache()
    print("running inference_e2e.py")
    os.system(
        f"python /home/hehaorui/code/Amphion/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output_dir}/recon/mel'} --output_dir={f'{args.output_dir}/recon/wav'} --checkpoint_file=/mnt/data2/wangyuancheng/ns2_ckpts/bigvgan/g_00490000 --gpu {args.cuda_id}"
    )
    os.system(
        f"python /home/hehaorui/code/Amphion/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output_dir}/target/mel'} --output_dir={f'{args.output_dir}/target/wav'} --checkpoint_file=/mnt/data2/wangyuancheng/ns2_ckpts/bigvgan/g_00490000 --gpu {args.cuda_id}"
    )
    os.system(
        f"python /home/hehaorui/code/Amphion/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output_dir}/source/mel'} --output_dir={f'{args.output_dir}/source/wav'} --checkpoint_file=/mnt/data2/wangyuancheng/ns2_ckpts/bigvgan/g_00490000 --gpu {args.cuda_id}"
    )
    os.system(
        f"python /home/hehaorui/code/Amphion/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output_dir}/prompt/mel'} --output_dir={f'{args.output_dir}/prompt/wav'} --checkpoint_file=/mnt/data2/wangyuancheng/ns2_ckpts/bigvgan/g_00490000 --gpu {args.cuda_id}"
    )
    with torch.device(f"cuda:{cuda_id}"):
        torch.cuda.empty_cache()
    print("running vc_test.py")
    os.system(f"python /home/hehaorui/code/Amphion/models/tts/vc/vc_test.py -r={f'{args.output_dir}/target/wav'} -d={f'{args.output_dir}/recon/wav'} --gpu {args.cuda_id}")

if __name__ == "__main__":
    main()