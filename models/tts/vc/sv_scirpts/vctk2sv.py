import itertools
import random
import os

# 初始化字典来存储说话者信息
def read_speaker_info(info_file_path):
    speakers_info = {}
    with open(info_file_path, 'r') as file:
        next(file)  # 跳过标题行
        for line in file:
            parts = line.split()
            speaker_id, gender = parts[0], parts[2]
            speakers_info[speaker_id] = gender
    return speakers_info
# 数据集的根目录

def get_data(audio_path, speaker_info):
    extracted_data = []
    for speaker_id in os.listdir(audio_path):
        speaker_dir = os.path.join(audio_path, speaker_id)
        if os.path.isdir(speaker_dir):
            for audio_file in os.listdir(speaker_dir):
                if audio_file.endswith("_mic1.flac"):
                    # 构建完整的音频文件路径
                    audio_file_path = os.path.join(speaker_dir, audio_file)
                    # 添加到提取的数据列表中
                    extracted_data.append({
                        'audio_file': audio_file.split('.')[0],
                        'speaker_id': speaker_id,
                        'gender': speaker_info.get(speaker_id, "Unknown"),
                        'file_path': audio_file_path
                    })
    print("Data size", len(extracted_data))
    return extracted_data



def generate_balanced_utterance_pairs(filtered_data):
    # Group utterances by gender
    gender_groups = {'M': [], 'F': []}
    for utt in filtered_data:
        gender = utt['gender']
        gender_groups[gender].append(utt)

    # Generate all possible same-gender pairs
    pairs_label_1 = []
    pairs_label_0 = []
    for gender in gender_groups:
        for utt1, utt2 in itertools.combinations(gender_groups[gender], 2):
            label = 1 if utt1['speaker_id'] == utt2['speaker_id'] else 0
            pair = f"{label} {utt1['file_path']} {utt2['file_path']}"
            if label == 1:
                pairs_label_1.append(pair)
            else:
                pairs_label_0.append(pair)

    print(f"Total pairs with label 1: {len(pairs_label_1)}")
    print(f"Total pairs with label 0: {len(pairs_label_0)}")
    # Randomly select label 0 pairs to match the number of label 1 pairs
    pairs_label_0 = random.sample(pairs_label_0, len(pairs_label_1))
    # Combine and shuffle
    balanced_pairs = pairs_label_1 + pairs_label_0
    random.shuffle(balanced_pairs)
    print(f"Total pairs with label 1: {len(pairs_label_1)}")
    print(f"Total pairs with label 0: {len(pairs_label_0)}")
    print("Total balanced pairs", len(balanced_pairs))
    return balanced_pairs

# 使用示例
txt_file = '/mnt/data2/hehaorui/vctk_sv_test.txt'  # 您想要创建的 TXT 文件的路径
dataset_path = "/mnt/data2/hehaorui/VCTK"
speaker_info_file_path = os.path.join(dataset_path, "speaker-info.txt")
audio_path = os.path.join(dataset_path, "wav48_silence_trimmed")
speaker_info = read_speaker_info(speaker_info_file_path)
data = get_data(audio_path,speaker_info)
data = random.choices(data, k = int(0.25 * len(data)))

# 统计 speaker 数量
unique_speakers = len(set(utt['speaker_id'] for utt in data))
print(f"Number of unique speakers: {unique_speakers}")

# 生成 utterance 对
pairs = generate_balanced_utterance_pairs(data)
pairs = random.choices(pairs, k=int(len(pairs) * 0.2))
print(f"Number of unique pairs: {len(pairs)}")

with open(txt_file, 'w', encoding='utf-8') as file:
    for pair in pairs:
        file.write(pair + "\n")
