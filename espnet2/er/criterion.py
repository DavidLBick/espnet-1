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
        mean_gt = torch.mean(ground_truth, 0)
        mean_pred = torch.mean(prediction, 0)
        var_gt = torch.var(ground_truth, 0)
        var_pred = torch.var(prediction, 0)
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        denominator = var_gt + var_pred + (mean_gt - mean_pred) ** 2
        cov = torch.mean(v_pred * v_gt)
        numerator = 2 * cov
        ccc = numerator / denominator
        return 1 - ccc, ccc
