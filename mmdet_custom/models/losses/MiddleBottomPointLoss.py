import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmdet.models.builder import MODELS
from .utils import weighted_loss
from typing import Optional


def get_worldcoord_from_imagecoord(imagecoord: Tensor, proj_mat: Tensor) -> Tensor:
    """Convert image coordinates to world coordinates using the projection matrix.

    Args:
        imagecoord (Tensor): Image coordinates of shape [N, 2].
        proj_mat (Tensor): Projection matrix of shape [4, 4].

    Returns:
        Tensor: World coordinates of shape [N, 2].
    """
    proj_mat = torch.linalg.inv(torch.cat((proj_mat[:, :2], proj_mat[:, 3:4]), dim=1)).float()
    image_coord = torch.cat((imagecoord, torch.ones((1, imagecoord.shape[1]), device=imagecoord.device)), dim=0)
    world_coord = torch.matmul(proj_mat, image_coord)
    world_coord = world_coord[:2, :] / world_coord[2,:]
    coord_x, coord_y = world_coord.unbind(dim=0)
    grid_x = (coord_x + 300) / 2.5
    grid_y = (coord_y + 900) / 2.5
    return torch.stack([grid_x, grid_y], dim=0).t()

def piecewise_weight(iteration):
    if iteration >= 10:
        return 1
    elif iteration >= 5:
        return 1.0
    else:
        return 1.0

# @weighted_loss
def MiddleBottomPointLoss(preds: Tensor, targets: Tensor, proj_mat: Optional[Tensor]):
    # Calculate middle bottom points for both predictions and targets
    middle_bottom_preds = torch.stack([(preds[:, 0] + preds[:, 2]) / 2, preds[:, 3]], dim=0)
    middle_bottom_targets = torch.stack([(targets[:, 0] + targets[:, 2]) / 2, targets[:, 3]], dim=0)

    # Transform to world coordinates
    world_middle_bottom_preds = get_worldcoord_from_imagecoord(middle_bottom_preds, proj_mat)
    world_middle_bottom_targets = get_worldcoord_from_imagecoord(middle_bottom_targets, proj_mat)
    # # Normalize world_middle_bottom_targets
    # norm_world_middle_bottom_targets = F.normalize(world_middle_bottom_targets, dim=1)
    # norm_world_middle_bottom_preds = F.normalize(world_middle_bottom_preds, dim=1)
    # # # Standardize the world coordinates
    # min_target = torch.min(world_middle_bottom_targets, dim=0, keepdim=True).values
    max_target = torch.max(world_middle_bottom_targets, dim=0, keepdim=True).values
    standardized_preds = world_middle_bottom_preds / (max_target + 1e-6)
    standardized_targets = world_middle_bottom_targets / (max_target + 1e-6)

    # print('standardized_preds: ', standardized_preds)
    # print('standardized_targets: ', standardized_targets)
    # print('---------------------------------------------')

    # Compute the absolute difference between standardized world coordinates
    # distances = torch.abs(world_middle_bottom_preds - world_middle_bottom_targets)
    distances = torch.abs(standardized_preds - standardized_targets)

    # print("distances: ", distances)
    # print('---------------------------------------------')
    # # Compute the loss as the mean of these distances
    loss = torch.sum(distances, dim=1)
    loss = torch.mean(loss)
    return loss
    # return distances
    
@weighted_loss
def l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L1 loss.

    Args:
        pred (Tensor): The prediction.
        target (Tensor): The learning target of the prediction.

    Returns:
        Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss 
   
@MODELS.register_module()    
class Custom_L1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self,
                 reduction: str = 'mean',
                 loss_weight_1: float = 1.0,
                 loss_weight_2: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_weight_1 = loss_weight_1
        self.loss_weight_2 = loss_weight_2

    def forward(self,
                pred: Tensor,
                pred_proj: Tensor,
                target: Tensor,
                target_proj: Tensor,
                weight: Optional[Tensor] = None,
                iteration = 0,
                proj_mat: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning target of the prediction.
            weight (Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox_1 = self.loss_weight_1 * l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        # print(loss_bbox_1)
        # loss_bbox_2 = self.loss_weight * MiddleBottomPointLoss(
        #     pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        # print(piecewise_weight(iteration))
        # if piecewise_weight(iteration) > 0:
        if False:
            loss_bbox_2 = self.loss_weight_2 * MiddleBottomPointLoss(
                pred_proj, target_proj, proj_mat)
            # loss_bbox_2 = piecewise_weight(iteration) * MiddleBottomPointLoss(
            #     pred_proj, target_proj, proj_mat=proj_mat, reduction=reduction, avg_factor=avg_factor)
            return loss_bbox_2 + loss_bbox_1
        else:
            return loss_bbox_1