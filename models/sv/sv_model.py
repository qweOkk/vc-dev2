import torch
import torch.nn as nn
from models.tts.vc.ns2_uniamphion import UniAmphionVC
from models.tts.vc.vc_loss import AMSoftmaxLoss
from safetensors.torch import load_model

import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F


class X_Vector(nn.Module):
    def __init__(self, nOut=512, p_dropout=0.1):
        super(X_Vector, self).__init__()
        self.instancenorm = nn.InstanceNorm1d(80)

        self.tdnn1 = nn.Conv1d(in_channels=80, out_channels=512, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn_tdnn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn_tdnn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(3000, 512)
        self.bn_fc1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        self.fc2 = nn.Linear(512, 512)
        self.bn_fc2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc2 = nn.Dropout(p=p_dropout)

        self.fc3 = nn.Linear(512, nOut)

    def forward(self, x):
        x = self.instancenorm(x)
        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
        x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(stats))))
        x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x


class sv_model(nn.Module):
    def __init__(self, config, num_speakers, vc_model_path):
        super().__init__() 
        self.vc_model = UniAmphionVC(config)
        print(f"Loading VC model from {vc_model_path}")
        load_model(self.vc_model, vc_model_path)

        self.num_speakers = num_speakers
        self.speaker_encoder = self.vc_model.reference_encoder
        # freeze the speaker encoder
        for param in self.speaker_encoder.parameters():
            param.requires_grad = False
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 192)
            )
        self.AMSoftmax = AMSoftmaxLoss(192, self.num_speakers)

    def forward(self, x, x_mask, label):
        with torch.no_grad():
            self.speaker_encoder.eval()
            x, _= self.speaker_encoder(x_ref=x, key_padding_mask=x_mask)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        loss = self.AMSoftmax(x, label)
        return loss
    
    @torch.no_grad()
    def sv_forward(self, x, x_mask):
        with torch.no_grad():
            self.speaker_encoder.eval()
            x, _= self.speaker_encoder(x_ref=x, key_padding_mask=x_mask)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x
