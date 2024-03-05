import torch
import torch.nn as nn


class threshold_loss(nn.Module):
    def __init__(self, config, **kwargs):
        super(threshold_loss, self).__init__()
        self.threshold = kwargs['threshold']

    def forward(self, similarity_scores, labels):
        '''
        if label == 1, similarity score should be larger than threshold, loss = -similarity_score
        if label == 0, similarity score should be smaller than threshold, loss = similarity_score
        '''
        labels = torch.where(labels == 0, torch.tensor(-1).to(labels.device), labels)
        loss = -1 * labels * (similarity_scores - self.threshold)
        return loss
