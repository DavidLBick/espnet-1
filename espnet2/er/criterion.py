"""
This file implements the criteria for continuous emotion recognition

"""


from typing import Union

import torch


class CCCLoss(torch.nn.Module):
    """
    Definition of the Concordance Correlation Coefficient (CCC) loss function
    """

    def __init__(self):
        super(CCCLoss, self).__init__()
        self.mean = torch.mean
        self.var = torch.var
        self.sum = torch.sum
        self.sqrt = torch.sqrt
        self.std = torch.std

    def forward(
        self, prediction: torch.Tensor, ground_truth: torch.Tensor
    ) -> Union[torch.Tensor, float]:
        """
        Args:
            prediction: (batch,)
            ground_truth: (batch,)
        Returns:
            loss: torch.Tensor 
            ccc: float- Concordance Correlation Coefficient
        """
        mean_gt = self.mean(ground_truth, 0)
        mean_pred = self.mean(prediction, 0)
        var_gt = self.var(ground_truth, 0)
        var_pred = self.var(prediction, 0)
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        cor = self.sum(v_pred * v_gt) / (
            self.sqrt(self.sum(v_pred**2)) * self.sqrt(self.sum(v_gt**2))
        )
        sd_gt = self.std(ground_truth)
        sd_pred = self.std(prediction)
        numerator = 2 * cor * sd_gt * sd_pred
        denominator = var_gt + var_pred + (mean_gt - mean_pred) ** 2
        ccc = numerator / denominator

        return 1 - ccc, ccc
