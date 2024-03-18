from models.codec.ns3_codec import FACodecEncoderV2, FACodecDecoderV2
import argparse
import torch
import numpy as np
import torch
from tqdm import tqdm
import librosa
import os
import json
import soundfile as sf

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

def load_facodec():
    fa_encoder_v2 = FACodecEncoderV2(
    ngf=32,
    up_ratios=[2, 4, 5, 5],
    out_channels=256,
    )

    fa_decoder_v2 = FACodecDecoderV2(in_channels=256,
        upsample_initial_channel=1024,
        ngf=32,
        up_ratios=[5, 5, 4, 2],
        vq_num_q_c=2,
        vq_num_q_p=1,
        vq_num_q_r=3,
        vq_dim=256,
        codebook_dim=8,
        codebook_size_prosody=10,
        codebook_size_content=10,
        codebook_size_residual=10,
        use_gr_x_timbre=True,
        use_gr_residual_f0=True,
        use_gr_residual_phone=True)


    encoder_v2_ckpt = "/mnt/data2/hehaorui/ckpt/facodec/ns3_facodec_encoder_v2.bin"
    decoder_v2_ckpt = "/mnt/data2/hehaorui/ckpt/facodec/ns3_facodec_decoder_v2.bin"

    fa_encoder_v2.load_state_dict(torch.load(encoder_v2_ckpt))
    fa_decoder_v2.load_state_dict(torch.load(decoder_v2_ckpt))
    return fa_encoder_v2, fa_decoder_v2


def facodec_vc_inference(wav_a, wav_b, fa_encoder_v2, fa_decoder_v2):
    with torch.no_grad():
        enc_out_a = fa_encoder_v2(wav_a)
        prosody_a = fa_encoder_v2.get_prosody_feature(wav_a)
        enc_out_b = fa_encoder_v2(wav_b)
        prosody_b = fa_encoder_v2.get_prosody_feature(wav_b)

        _, vq_id_a, _, _, _ = fa_decoder_v2(
            enc_out_a, prosody_a, eval_vq=False, vq=True
        )
        _, _, _, _, spk_embs_b = fa_decoder_v2(
            enc_out_b, prosody_b, eval_vq=False, vq=True
        )

        vq_post_emb_a_to_b = fa_decoder_v2.vq2emb(vq_id_a, use_residual=False)
        recon_wav_a_to_b = fa_decoder_v2.inference(vq_post_emb_a_to_b, spk_embs_b)
        # wav a is source, wav b is reference
        return recon_wav_a_to_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of dataloader workers."
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

    cuda_id = args.cuda_id
    args.local_rank = torch.device(f"cuda:{cuda_id}")
    print("local rank", args.local_rank)

    zero_shot_json_file_path = args.zero_shot_json_file_path
    print("zero_shot_json_file_path", zero_shot_json_file_path)
    with torch.cuda.device(args.local_rank):
        torch.cuda.empty_cache()

    print("loading model")
    encoder, decoder = load_facodec()
    encoder.cuda(args.local_rank).eval()
    decoder.cuda(args.local_rank).eval()
    print("model loaded")


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

    # if output_dir exists, delete it
    if os.path.exists(args.output_dir):
        os.system(f"rm -r {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f"{args.output_dir}/recon/wav", exist_ok=True)
    os.makedirs(f"{args.output_dir}/target/wav", exist_ok=True)
    os.makedirs(f"{args.output_dir}/source/wav", exist_ok=True)
    os.makedirs(f"{args.output_dir}/prompt/wav", exist_ok=True)


    temp_id = 0
    for utt_id, utt in tqdm(utt_dict.items()):
        temp_id += 1

        # source is the input
        wav_path = utt["source_speech"]
        wav, _ = librosa.load(wav_path, sr=16000)
        wav = np.pad(wav, (0, 1600 - len(wav) % 1600))
        # to numpy.ndarray
        audio = torch.from_numpy(wav).float()
        audio = audio.unsqueeze(0).unsqueeze(0).to(args.local_rank)
        
        # target is the ground truth
        tgt_wav_path = utt["target_speech"]
        tgt_wav, _ = librosa.load(tgt_wav_path, sr=16000)
        tgt_wav = np.pad(tgt_wav, (0, 1600 - len(tgt_wav) % 1600))
        tgt_audio = torch.from_numpy(tgt_wav).float()
        tgt_audio = tgt_audio.unsqueeze(0).unsqueeze(0).to(args.local_rank)

        # prompt is the reference
        ref_wav_path = utt["prompt_speech"]
        ref_wav,_ = librosa.load(ref_wav_path, sr=16000)
        ref_wav = add_noise(ref_wav)
        ref_wav = np.pad(ref_wav, (0, 1600 - len(ref_wav) % 1600))

        ref_audio = torch.from_numpy(ref_wav).float()
        ref_audio = ref_audio.unsqueeze(0).unsqueeze(0).to(args.local_rank)

        recon_audio = facodec_vc_inference(audio, ref_audio, encoder, decoder)
        
        recon_audio = recon_audio.squeeze().cpu().numpy()
        # save source/reference/target/recon to args.output_dir
        recon_path = f"{args.output_dir}/recon/wav/recon_{utt_id}_generated_e2e.wav"
        prompt_path = f"{args.output_dir}/prompt/wav/prompt_{utt_id}_generated_e2e.wav"
        ref_path = f"{args.output_dir}/target/wav/target_{utt_id}_generated_e2e.wav"
        source_path = f"{args.output_dir}/source/wav/source_{utt_id}_generated_e2e.wav"

        # write to file with sf
        sf.write(recon_path, recon_audio, 16000)
        sf.write(prompt_path, ref_wav, 16000)
        sf.write(ref_path, tgt_wav, 16000)
        sf.write(source_path, wav, 16000)

    with torch.cuda.device(args.local_rank):
        torch.cuda.empty_cache()
    print("running vc_test.py")
    os.system(f"python /home/hehaorui/code/Amphion/models/tts/vc/vc_test.py -r={f'{args.output_dir}/target/wav'} -d={f'{args.output_dir}/recon/wav'} --gpu {args.cuda_id}")

if __name__ == "__main__":
    main()