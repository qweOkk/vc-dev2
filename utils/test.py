import os
import tarfile
from tqdm import tqdm

# 输入路径和文件名
data_dir = "/mnt/data3/share/voxceleb/voxceleb1/wav/"
split_file = "/mnt/data3/share/voxceleb/voxceleb1/iden_split.txt"
tar_file = "/mnt/data3/share/voxceleb/voxceleb1/voxceleb_test.tar"

# 打开iden_split.txt文件并读取split=3的文件路径
split_3_files = []
with open(split_file, 'r') as f:
    for line in f:
        split, file_path = line.strip().split(' ', 1)
        if split == '3':
            split_3_files.append(file_path)

# 打印split=3的文件个数
print("Number of files with split=3:", len(split_3_files))

# 创建tar包并将split=3的文件添加进去
with tarfile.open(tar_file, 'w') as tar:
    # 迭代文件列表并添加到tar包
    for file_path in tqdm(split_3_files, desc="Progress"):
        # 获取文件的绝对路径
        abs_file_path = os.path.join(data_dir, file_path)
        # 将文件添加到tar包中
        tar.add(abs_file_path, arcname=file_path)

print("Tarball created successfully at:", tar_file)
