import os
import json

def create_metadata(root_dir, split, start_index):
    metadata = []
    
    clean_speech_dir = os.path.join(root_dir, 'wavs', split, 'CleanSpeech_' + split + 'ing')
    noisy_speech_dir = os.path.join(root_dir, 'wavs', split, 'NoisySpeech_' + split + 'ing')
    noise_dir = os.path.join(root_dir, 'wavs', split, 'Noise_' + split + 'ing')

    for index, clean_file in enumerate(os.listdir(clean_speech_dir)):
        uid = clean_file.split('.')[0]
        clean_path = os.path.join(clean_speech_dir, clean_file)
        if split == 'train':
            noise_paths = [os.path.join(noise_dir, f'{uid}_SNRdb_{i:.1f}.wav') for i in [0.0, 10.0, 20.0, 30.0, 40.0]]
            noisy_speech_paths = [os.path.join(noisy_speech_dir, f'{uid}_SNRdb_{i:.1f}.wav') for i in [0.0, 10.0, 20.0, 30.0, 40.0]]
        else:
            noise_paths = [os.path.join(noise_dir, f'{uid}_SNRdb_{i:.1f}.wav') for i in [2.0, 12.0, 22.0, 32.0, 42.0]]
            noisy_speech_paths = [os.path.join(noisy_speech_dir, f'{uid}_SNRdb_{i:.1f}.wav') for i in [2.0, 12.0, 22.0, 32.0, 42.0]]         

        # Check if paths are valid
        paths_to_check = [clean_path] + noise_paths + noisy_speech_paths
        invalid_paths = [path for path in paths_to_check if not os.path.isfile(path)]

        if invalid_paths:
            for path in invalid_paths:
                print(path)

        entry = {
            "Task": "SE",
            "Dataset": "LibriTTS_SE",
            "Split": split,
            "Index": index + start_index,
            "Uid": uid + '_' + split,
            "Path": clean_path,
            "NoisePath": noise_paths,
            "NoisySpeechPath": noisy_speech_paths,
        }

        metadata.append(entry)

    return metadata, len(os.listdir(clean_speech_dir))

def main():
    root_dir = '/nvme/uniamphion/se/Libritts_SE/'
    metadata_test, cur_index = create_metadata(root_dir, 'test', 0)
    metadata_train, cur_index = create_metadata(root_dir, 'train', cur_index)
    #metadata = metadata_test
    metadata = metadata_test + metadata_train

    with open('/nvme/uniamphion/se/Libritts_SE/metadata.json', 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

if __name__ == "__main__":
    main()

