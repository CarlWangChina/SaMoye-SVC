import torch.nn as nn
from torch import Tensor


class DiffusionNoiseLoss(nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.loss_type = loss_type
        if self.loss_type == 'l1':
            self.loss = nn.L1Loss(reduction='none')
        elif self.loss_type == 'l2':
            self.loss = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError()

    @staticmethod
    def _mask_nonpadding(x_recon, noise, nonpadding=None):
        if nonpadding is not None:
            nonpadding = nonpadding.transpose(1, 2).unsqueeze(1)
            return x_recon * nonpadding, noise * nonpadding
        else:
            return x_recon, noise

    def _forward(self, x_recon, noise):
        return self.loss(x_recon, noise)

    def forward(self, x_recon: Tensor, noise: Tensor, nonpadding: Tensor = None) -> Tensor:
        """
        :param x_recon: [B, 1, M, T]
        :param noise: [B, 1, M, T]
        :param nonpadding: [B, T, M]
        """
        x_recon, noise = self._mask_nonpadding(x_recon, noise, nonpadding)
        return self._forward(x_recon, noise).mean()
