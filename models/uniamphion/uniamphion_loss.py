import torch
import torch.nn.functional as F

def log_dur_loss(dur_pred_log, dur_target, mask, loss_type="l1"):
    dur_target_log = torch.log(1 + dur_target)
    if loss_type == "l1":
        loss = F.l1_loss(
            dur_pred_log, dur_target_log, reduction="none"
        ).float() * mask.to(dur_target.dtype)
    elif loss_type == "l2":
        loss = F.mse_loss(
            dur_pred_log, dur_target_log, reduction="none"
        ).float() * mask.to(dur_target.dtype)
    else:
        raise NotImplementedError()
    loss = loss.sum() / (mask.to(dur_target.dtype).sum())
    return loss


def log_pitch_loss(pitch_pred_log, pitch_target, mask, loss_type="l1"):
    # pitch_target_log = torch.log(pitch_target)
    pitch_target_log = torch.log(1 + pitch_target)
    if loss_type == "l1":
        loss = F.l1_loss(
            pitch_pred_log, pitch_target_log, reduction="none"
        ).float() * mask.to(pitch_target.dtype)
    elif loss_type == "l2":
        loss = F.mse_loss(
            pitch_pred_log, pitch_target_log, reduction="none"
        ).float() * mask.to(pitch_target.dtype)
    else:
        raise NotImplementedError()
    loss = loss.sum() / (mask.to(pitch_target.dtype).sum() + 1e-8)
    return loss


def diff_loss(pred, target, mask, loss_type="l1"):
    # pred: (B, T, d)
    # target: (B, T, d)
    # mask: (B, T)
    if loss_type == "l1":
        loss = F.l1_loss(pred, target, reduction="none").float() * (
            mask.to(pred.dtype).unsqueeze(-1)
        )
    elif loss_type == "l2":
        loss = F.mse_loss(pred, target, reduction="none").float() * (
            mask.to(pred.dtype).unsqueeze(-1)
        )
    else:
        raise NotImplementedError()
    loss = (torch.mean(loss, dim=-1)).sum() / (mask.to(pred.dtype).sum())
    return loss
