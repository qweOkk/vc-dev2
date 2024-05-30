import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram

# 加载音频文件
audio_path = '/home/hehaorui/code/Amphion/p282_005.wav'
sample_rate, samples = wavfile.read(audio_path)

# 添加随机噪声
np.random.seed(42)  # 设置随机种子以确保可重复性
noise = np.random.normal(0, 100, len(samples))
samples_with_noise = samples + noise

# 可视化波形, 
plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0, len(samples_with_noise) / sample_rate, num=len(samples_with_noise)), samples_with_noise, color='red')
plt.title('Waveform with Noise')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.savefig('waveform_with_noise.png', dpi=300)  # 保存为png格式的高分辨率图像
plt.show()

# 计算并可视化Mel频谱
frequencies, times, Sxx = spectrogram(samples_with_noise, sample_rate)
plt.figure(figsize=(12, 6))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.title('Mel Spectrogram with Noise')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Intensity [dB]')
plt.tight_layout()
plt.savefig('mel_spectrogram_with_noise.png', dpi=300)  # 保存为png格式的高分辨率图像
plt.show()
