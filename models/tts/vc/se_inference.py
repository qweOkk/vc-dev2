import argparse
import torch
import numpy as np
import torch
from safetensors.torch import load_model
import librosa
from utils.util import load_config
import os
from models.tts.vc.vc_trainer import VCTrainer, mel_spectrogram, extract_world_f0
from models.tts.vc.ns2_uniamphion import UniAmphionVC
from models.tts.vc.hubert_kmeans import HubertWithKmeans

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
        "--cuda_id", 
        type=int, 
        default=7, 
        help="Cuda id for training."
    )

    args = parser.parse_args()
    cfg = load_config(args.config)

    # Model saving dir
    cuda_id = args.cuda_id
    args.local_rank = torch.device(f"cuda:{cuda_id}")
    print("local rank", args.local_rank)

    ckpt_path = args.checkpoint_path
    print("ckpt_path", ckpt_path)

    # Load model
    model = UniAmphionVC(cfg=cfg.model, use_speaker = False, speaker_num = 12954)
    print("loading model")
    load_model(model, ckpt_path)
    print("model loaded")
    model.cuda(args.local_rank)
    model.eval()

    # paths to be modified
    HubertWithKmeans_path = "/mnt/data3/hehaorui/ckpt/mhubert"
    wav_path = "/mnt/data2/hehaorui/VoxCeleb1/wav/id10407/eD0bE59GgGU/00001.wav"
    vocoder_path = "/mnt/data2/wangyuancheng/ns2_ckpts/bigvgan/g_00490000"

    os.makedirs(f"{args.output_dir}/recon/mel", exist_ok=True)
    os.makedirs(f"{args.output_dir}/source/mel", exist_ok=True)

    # Load w2v
    w2v = HubertWithKmeans(
            checkpoint_path=HubertWithKmeans_path + "/mhubert_base_vp_en_es_fr_it3.pt",
            kmeans_path=HubertWithKmeans_path + "/mhubert_base_vp_en_es_fr_it3_L11_km1000.bin",
        )
    w2v = w2v.to(device=args.local_rank)
    w2v.eval()

    wav, _ = librosa.load(wav_path, sr=16000)
    wav = np.pad(wav, (0, 1600 - len(wav) % 1600))
    audio = torch.from_numpy(wav).to(args.local_rank)
    audio = audio[None, :]

    with torch.no_grad():
        ref_mel = mel_spectrogram(
            audio,
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
        out_mel = model.inference(
            content_feature=content_feature,
            pitch=pitch,
            x_ref=ref_mel,
            x_ref_mask=ref_mask,
            inference_steps=200,
            sigma=1.2,
        ) # here mel is the mel spectrogram of the enhanced audio

        # source_mel : mel spectrogram of the noisy audio
        # out_mel : mel spectrogram of the enhanced audio
        assert source_mel.detach().cpu().numpy().shape ==out_mel.transpose(1, 2).detach().cpu().numpy().shape, "source_mel and out_mel should have the same shape"
        recon_path = f"{args.output_dir}/recon/mel/recon.npy"
        source_path = f"{args.output_dir}/source/mel/source.npy"

        # save the mel spectrogram of the enhanced audio and the mel spectrogram of the noisy audio
        np.save(recon_path, out_mel.transpose(1, 2).detach().cpu().numpy())
        np.save(source_path, source_mel.detach().cpu().numpy())
        # mel to wav
        print("running inference_e2e.py")
        os.system(
            f"python /home/hehaorui/code/Amphion/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output_dir}/source/mel'} --output_dir={f'{args.output_dir}/source/wav'} --checkpoint_file={vocoder_path} --gpu {args.cuda_id}"
        )
        os.system(
            f"python /home/hehaorui/code/Amphion/BigVGAN/inference_e2e.py --input_mels_dir={f'{args.output_dir}/recon/mel'} --output_dir={f'{args.output_dir}/recon/wav'} --checkpoint_file={vocoder_path} --gpu {args.cuda_id}"
        )
        print("inference_e2e.py finished")

if __name__ == "__main__":
    main()