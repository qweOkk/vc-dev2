import torch.nn as nn


class cosine_similarity_decision(nn.Module):
    def __init__(self, config, **kwargs):
        super(cosine_similarity_decision, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.threshold = kwargs['threshold']

    def forward(self, x1, x2):
        cosine_similarity = self.cos(x1, x2)
        decisions = cosine_similarity > self.threshold
        return decisions
