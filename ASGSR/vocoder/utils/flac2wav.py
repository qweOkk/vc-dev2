import os

# for root, dir, files in os.walk('/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_eval/flac'):
#     for file in files:
#         if file.endswith('flac'):
#             os.system('ffmpeg -i {} {}'.format(os.path.join(root, file), os.path.join(root, file.replace('flac', 'wav'))))

for root, dir, files in os.walk('/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_train/flac'):
    for file in files:
        if file.endswith('flac'):
            os.system('ffmpeg -i {} {}'.format(os.path.join(root, file), os.path.join(root, file.replace('flac', 'wav'))))

# for root, dir, files in os.walk('/mntnfs/lee_data1/wangli/ASVspoof2019/PA/ASVspoof2019_PA_dev/flac'):
#     for file in files:
#         if file.endswith('flac'):
#             os.system('ffmpeg -i {} {}'.format(os.path.join(root, file), os.path.join(root, file.replace('flac', 'wav'))))
