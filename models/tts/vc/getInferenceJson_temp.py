import os
import json
import random

# Define the directories
folder = '/mnt/data2/hehaorui/vc_test/Celeb'
gt_dir = folder + '/gt'
prompt_dir = folder + '/prompt'
source_dir =  folder + '/source'

# List files in each directory and sort them
gt_files = sorted(os.listdir(gt_dir))
prompt_files = sorted(os.listdir(prompt_dir))
print(prompt_files)
source_files = sorted(os.listdir(source_dir))

random.seed(0)
# length equaal to prompt files
gt_files = random.sample(gt_files, len(prompt_files))
source_files = random.sample(source_files, len(prompt_files))
# Initialize the list for test cases
test_cases = []

# Iterate over files in all directories simultaneously
for source_file, gt_file, prompt_file in zip(source_files, gt_files, prompt_files):
    # Extract UIDs and check if the last three digits match
    source_uid = source_file.split('.')[0]
    gt_uid = gt_file.split('.')[0]
    # if source_uid[-3:] == gt_uid[-3:]:
        # Construct file paths
    source_wav_path = os.path.join(source_dir, source_file)
    target_wav_path = os.path.join(gt_dir, gt_file)
    prompt_wav_path = os.path.join(prompt_dir, prompt_file)

    # Create a dictionary for the test case
    test_case = {
        "uid": source_uid,  # or use gt_uid based on which is preferred
        "source_wav_path": source_wav_path,
        "target_wav_path": target_wav_path,
        "prompt_wav_path": prompt_wav_path
    }

    # Add the test case to the list
    test_cases.append(test_case)
assert len(test_cases) == len(prompt_files)

# Save the test cases to a JSON file
zero_shot_json_file_path = '/home/hehaorui/code/Amphion/egs/tts/VC/zero_shot_json_Celeb_new.json'
with open(zero_shot_json_file_path, 'w') as f:
    json.dump({"test_cases": test_cases}, f, indent=4)
