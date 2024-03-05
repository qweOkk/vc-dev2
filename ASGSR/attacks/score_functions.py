import torch.nn as nn


class cosine_similarity_score(nn.Module):
    def __init__(self, config, **kwargs):
        super(cosine_similarity_score, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, x1, x2):
        return self.cos(x1, x2)
