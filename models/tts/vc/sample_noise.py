import os
import random
import tarfile

# 噪音文件目录
noise_dir = '/home/hehaorui/code/Amphion/MS-SNSD/noise_train'  # 修改为实际路径

# 获取目录下的所有文件
all_files = os.listdir(noise_dir)

# 获取每种噪音类型的文件
noise_types = {}
for file in all_files:
    if file.endswith('.wav'):
        noise_type = file.split('_')[0]
        if noise_type not in noise_types:
            noise_types[noise_type] = []
        noise_types[noise_type].append(file)

# 随机选择每种噪音类型的一个文件
selected_files = [random.choice(files) for files in noise_types.values()]

# 创建 tar 包
with tarfile.open('selected_noise_files.tar', 'w') as tar:
    for file in selected_files:
        tar.add(os.path.join(noise_dir, file), arcname=file)

print('Tar package created successfully: selected_noise_files.tar')
