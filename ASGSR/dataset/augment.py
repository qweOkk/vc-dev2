import glob
import os
import random
import torch
from utils.io_utils import load_waveform_torch
from utils.audio_utils import pad_cut


class AugmentWAV(object):

    def __init__(self, musan_path, rir_path, max_len):

        self.max_len = max_len

        self.noisetypes = ['noise', 'speech', 'music']

        self.noisesnr = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
        self.numnoise = {'noise': [1, 1], 'speech': [3, 7], 'music': [1, 1]}
        self.noiselist = {}

        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

    def additive_noise(self, noisecat, audio):
        # clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        clean_db = 10 * torch.log10(torch.mean(audio ** 2) + 1e-4)

        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))

        noises = []

        for noise in noiselist:
            noiseaudio, _ = load_waveform_torch(noise)
            noiseaudio = pad_cut(noiseaudio, self.max_len)

            # noiseaudio = loadWAV(noise, self.max_len)
            noise_snr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])

            # noise_db = 10 * numpy.log10(numpy.mean(noiseaudio[0] ** 2) + 1e-4)
            noise_db = 10 * torch.log10(torch.mean(noiseaudio[0] ** 2) + 1e-4)

            # noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)
            noises.append(torch.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)

        # numpy: numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True) + audio
        return torch.sum(torch.cat(noises, dim=0), dim=0, keepdim=True) + audio

    def reverberate(self, audio):

        rir_file = random.choice(self.rir_files)

        # rir, fs = soundfile.read(rir_file)
        rir, fs = load_waveform_torch(rir_file)
        # rir = numpy.expand_dims(rir.astype(numpy.float), 0)
        rir = rir.unsqueeze(0)
        # numpy rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
        rir = rir / torch.sqrt(torch.sum(rir ** 2))
        # convolution: signal.convolve(audio, rir, mode='full')[:, :self.max_len]
        return pad_cut(torch.nn.functional.conv1d(audio, rir),self.max_len)
        # return signal.convolve(audio, rir, mode='full')[:, :self.max_len]

    def process(self, audio):
        augtype = random.randint(1, 4)
        # print("augtype", augtype)
        if augtype == 1:
            audio = self.reverberate(audio)
        elif augtype == 2:
            audio = self.additive_noise('music', audio)
        elif augtype == 3:
            audio = self.additive_noise('speech', audio)
        elif augtype == 4:
            audio = self.additive_noise('noise', audio)
        return audio
