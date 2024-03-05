
import pandas as pd
from models.tts.vc.sv_scirpts.sv_metrics import compute_pmiss_pfa_rbst,compute_eer,compute_c_norm

import torch
import torch.nn.functional as F

def get_sv_data(path):
    data = []
    with open(path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            data.append(parts)
    df = pd.DataFrame(data, columns=['Label', 'First', 'Second'])
    return df

def metric(scores, labels):
    p_target = 0.01
    c_miss = 1
    c_fa = 1
    fnr, fpr = compute_pmiss_pfa_rbst(scores, labels)
    eer, thres = compute_eer(fnr, fpr, scores)
    min_dcf = compute_c_norm(fnr, fpr, p_target=p_target, c_miss=c_miss, c_fa=c_fa)
    print("EER = {0:.3f}".format(100 * eer))
    print("minDCF = {:.3f}".format(min_dcf))


def get_batch_cos_sim(emb1, emb2):
    # 计算 A 和 B 的单位向量
    A_norm = F.normalize(emb1, p=2, dim=1)
    B_norm = F.normalize(emb2, p=2, dim=1)

    # 计算余弦相似度
    scores = torch.sum(A_norm * B_norm, dim=1)
    return scores