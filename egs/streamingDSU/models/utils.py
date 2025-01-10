
import joblib
import torch
import torch.nn as nn


class ApplyKmeans(object):
    def __init__(self, km_path, use_gpu):
        self.km_model = joblib.load(km_path)
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np**2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if use_gpu and torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        x = x.to(self.C.device)
        dist = (
            x.pow(2).sum(1, keepdim=True) - 2 * torch.matmul(x, self.C) + self.Cnorm
        )
        return dist.argmin(dim=1).cpu().numpy()


class CentroidLoss(nn.Module):
    def __init__(self, km_path, use_gpu):
        super(CentroidLoss, self).__init__()
        km = ApplyKmeans(km_path, use_gpu)
        self.C = km.C
    
    def forward(self, centroids, units, unit_lengths):
        # x is the output
        B = unit_lengths.shape

        loss = 0
        for b in range(B):
            unit_length = unit_lengths[b]
            true_centroids = torch.index_select(
                self.C, 0,
                units[b][:unit_length]
            ).reshape(1024, unit_length) # (1024, unit_length)
            loss += torch.abs(
                centroids[b][:unit_length].transpose(0, 1) - true_centroids
            ).sum(dim=0).mean()
        
        loss /= B
        return loss
