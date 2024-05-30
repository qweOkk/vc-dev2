import json
import os
import Levenshtein
import librosa
import nltk
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from modelscope.pipelines import pipeline  # run !pip install -U funasr modelscope
from modelscope.utils.constant import Tasks  # run !pip install -U funasr modelscope
import whisper  # pip install -U openai-whisper


def calculate_fid(target_wav, reference_wav):
    """
    Calculate the cosine similarity between emotion embeddings of two waveforms.
    """
    inference_pipeline = pipeline(
        task=Tasks.emotion_recognition,
        model="iic/emotion2vec_base",
        model_revision="v2.0.4",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    results = inference_pipeline([target_wav, reference_wav], granularity="utterance")
    rec_target = results[0]["feats"]  # (768,)
    rec_reference = results[1]["feats"]  # (768,)
    cos_sim_score = F.cosine_similarity(
        torch.tensor(rec_target), torch.tensor(rec_reference), dim=-1
    )
    return cos_sim_score.item()


def calculate_speaker_similarity(target_wav, reference_wav, device):
    """
    Extract acoustic embeddings and calculate cosine similarity.
    """
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "microsoft/wavlm-base-plus-sv"
    )
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
    if device == "cuda":
        print("Cuda available, conducting inference on GPU")
        model = model.to(device)

    inputs = feature_extractor(
        [target_wav, reference_wav], padding=True, return_tensors="pt"
    )

    if device == "cuda":
        for key in inputs.keys():
            inputs[key] = inputs[key].cuda("cuda")

    with torch.no_grad():
        embeddings = model(**inputs).embeddings
        embeddings = embeddings.cpu()
        cos_sim_score = F.cosine_similarity(embeddings[0], embeddings[1], dim=-1)

    return cos_sim_score.item()


def calculate_wer(transcript_text, target_text, device):
    """
    Calculate Word Error Rate (WER) between two texts.
    """
    hyp_words = nltk.word_tokenize(transcript_text.lower())
    ref_words = nltk.word_tokenize(target_text.lower())

    distance = Levenshtein.distance(ref_words, hyp_words)
    wer = distance / len(ref_words)
    return wer


if __name__ == "__main__":
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    whisper_model = whisper.load_model(".large-v2.pt") # load from local path

    # Load model [Need to change]
    tts_model = torch.load("./tts.pth", map_location=device)

    reference_folder = "Wave16k16bNormalized"
    output_folder = "gen_data"

    # Load json file
    with open("./librispeech_ref_dur_3_test_full_with_punc_wdata.json", "r") as f:
        json_data = f.read()
    data = json.loads(json_data)
    test_data = data["test_cases"]

    wer_scores = []
    similarity_scores = []
    fid_scores = []
    nltk.download("punkt")
    for wav_info in tqdm(test_data):
        wav_path = wav_info["wav_path"].split("/")[-1]
        reference_path = os.path.join(reference_folder, wav_path)
        assert os.path.exists(reference_path), f"File {reference_path} not found"

        reference_wav, _ = librosa.load(reference_path, sr=16000)
        source_text = wav_info["text"]
        target_text = wav_info["target_text"]

        output_file_name = wav_info["uid"] + ".wav"
        output_path = os.path.join(output_folder, output_file_name)

        # Run TTS based on own model [Need to change]
        output_wav = tts_model(
            source_text=source_text,
            target_text=target_text,
            reference_wav=reference_wav,
            language="en",
        )

        # WER:
        transcript_text = whisper_model.transcribe(output_wav)["text"]
        wer = calculate_wer(transcript_text, target_text, device)
        # SIM-O
        sim_o = calculate_speaker_similarity(output_wav, reference_wav, device)
        # FID:
        fid = calculate_fid(output_wav, reference_wav)

        wer_scores.append(wer)
        similarity_scores.append(sim_o)
        fid_scores.append(fid)

    print(f"WER: {np.mean(wer_scores)}")
    print(f"SIM-O: {np.mean(similarity_scores)}")
    print(f"FID: {np.mean(fid_scores)}")
