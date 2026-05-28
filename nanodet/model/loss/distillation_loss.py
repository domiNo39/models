import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, feat_weight=1.0):
        super(DistillationLoss, self).__init__()
        self.feat_weight = feat_weight

    def forward(self, student_feats, teacher_feats):
        loss = 0
        for s_feat, t_feat in zip(student_feats, teacher_feats):
            if s_feat.shape != t_feat.shape:
                t_feat = F.interpolate(t_feat, size=s_feat.shape[-2:], mode='bilinear', align_corners=False)

            loss += F.mse_loss(s_feat, t_feat)

        return loss * self.feat_weight