# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn.functional as F

# from ..builder import BBOX_ASSIGNERS
# from ..iou_calculators import bbox_overlaps
# from .assign_result import AssignResult
# from .base_assigner import BaseAssigner
from mmdet.core import BaseAssigner, AssignResult, bbox_cxcywh_to_xyxy
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost


@BBOX_ASSIGNERS.register_module()
class SimCTOTAAssigner(BaseAssigner):
    """Computes matching between predictions and ground truth.

    Args:
        center_radius (int | float, optional): Ground truth center size
            to judge whether a prior is in center. Default 2.5.
        candidate_topk (int, optional): The candidate top-k which used to
            get top-k ious to calculate dynamic-k. Default 10.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 3.0.
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 center_radius=2.5,
                 candidate_topk=9,
                 iou_weight=3.0,
                 cls_weight=1.0):
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.reg_weight = iou_weight
        self.cls_weight = cls_weight
        self.cls_cost = build_match_cost(cls_cost)

    def assign(self,
               pred_scores,
               priors,
               decoded_ct,
               gt_ct,
               gt_labels,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Assign gt to priors using SimOTA. It will switch to CPU mode when
        GPU is out of memory.
        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            eps (float): A value added to the denominator for numerical
                stability. Default 1e-7.
        Returns:
            assign_result (obj:`AssignResult`): The assigned result.
        """
        assign_result = self._assign(pred_scores, priors, decoded_ct,
                                     gt_ct, gt_labels,
                                     gt_bboxes_ignore, eps)
        return assign_result

    def _assign(self,
                cls_pred,
                priors,
                decoded_ct,
                gt_ct,
                gt_labels,
                gt_bboxes_ignore=None,
                eps=1e-7):
        """Assign gt to priors using SimOTA.
        Args:
            pred_scores (Tensor): Classification scores of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Predicted bboxes, a 2D-Tensor with shape
                [num_priors, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            eps (float): A value added to the denominator for numerical
                stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        INF = 100000000
        num_gt = gt_ct.size(0)
        num_bboxes = decoded_ct.size(0)

        # assign 0 by default
        assigned_gt_inds = decoded_ct.new_full((num_bboxes, ),
                                                   0,
                                                   dtype=torch.long)
        # valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
        #     priors, gt_ct)
        valid_mask, is_in_boxes_and_center = self.get_in_gt_and_in_center_info(
            priors, gt_ct)
        valid_decoded_ct = decoded_ct[valid_mask]
        valid_cls_pred = cls_pred[valid_mask]
        num_valid = valid_decoded_ct.size(0)

        if num_gt == 0 or num_bboxes == 0 or num_valid == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = decoded_ct.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = decoded_ct.new_full((num_bboxes, ),
                                                          -1,
                                                          dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
        reg_cost = torch.cdist(valid_decoded_ct, gt_ct, p=1)/4
        # pairwise_ious = bbox_overlaps(valid_decoded_ct, gt_ct)
        # iou_cost = -torch.log(pairwise_ious + eps)

        # gt_onehot_label = (
        #     F.one_hot(gt_labels.to(torch.int64),
        #               pred_scores.shape[-1]).float().unsqueeze(0).repeat(
        #                   num_valid, 1, 1))

        # valid_cls_pred = valid_cls_pred.unsqueeze(1).repeat(1, num_gt, 1)
        # cls_cost = F.binary_cross_entropy(
        #     valid_pred_scores.sqrt_(), gt_onehot_label,
        #     reduction='none').sum(-1)
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        #     reduction='none').sum(-1)
        cost_matrix = (
            cls_cost * self.cls_weight + reg_cost * self.reg_weight)

        # matched_pred_ious, matched_gt_inds = \
        #     self.dynamic_k_matching(
        #         cost_matrix, reg_cost, num_gt, topk=self.candidate_topk)
        matched_pred_ious, matched_gt_inds = \
            self.dynamic_k_matching(
                cost_matrix, reg_cost, num_gt, topk=self.candidate_topk)
        # convert to AssignResult format
        assigned_gt_inds[valid_mask] = matched_gt_inds + 1
        assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
        assigned_labels[valid_mask] = gt_labels[matched_gt_inds].long()
        max_overlaps = assigned_gt_inds.new_full((num_bboxes, ),
                                                 -INF,
                                                 dtype=torch.float32)
        max_overlaps[valid_mask] = matched_pred_ious

        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)
    def get_in_gt_and_in_center_info(self, priors, gt_bboxes):
        num_gt = gt_bboxes.size(0)

        repeated_x = priors[:, 0].unsqueeze(1).repeat(1, num_gt)
        repeated_y = priors[:, 1].unsqueeze(1).repeat(1, num_gt)


        # is prior centers in gt centers
        gt_cxs = gt_bboxes[:,0]
        gt_cys = gt_bboxes[:,1]
        ct_box_l = gt_cxs - self.center_radius
        ct_box_t = gt_cys - self.center_radius
        ct_box_r = gt_cxs + self.center_radius
        ct_box_b = gt_cys + self.center_radius

        cl_ = repeated_x - ct_box_l
        ct_ = repeated_y - ct_box_t
        cr_ = ct_box_r - repeated_x
        cb_ = ct_box_b - repeated_y

        ct_deltas = torch.stack([cl_, ct_, cr_, cb_], dim=1)
        is_in_cts = ct_deltas.min(dim=1).values > 0
        is_in_cts_all = is_in_cts.sum(dim=1) > 0

        # in boxes or in centers, shape: [num_priors]
        is_in_gts_or_centers = is_in_cts_all

        # both in boxes and centers, shape: [num_fg, num_gt]
        is_in_boxes_and_centers = ( is_in_cts[is_in_gts_or_centers, :])
        return is_in_gts_or_centers, is_in_boxes_and_centers

    def dynamic_k_matching(self, cost, pairwise_ious, num_gt, topk):
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)
        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(topk, pairwise_ious.size(0))
        topk_ious, _ = torch.topk(pairwise_ious, candidate_topk, dim=0)
        # calculate dynamic k for each gt
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        prior_match_gt_mask = matching_matrix.sum(1) > 1
        if prior_match_gt_mask.sum() > 0:
            cost_min, cost_argmin = torch.min(
                cost[prior_match_gt_mask, :], dim=1)
            matching_matrix[prior_match_gt_mask, :] *= 0
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0
        # valid_mask[valid_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        matched_pred_ious = (matching_matrix *
                             pairwise_ious).sum(1)[fg_mask_inboxes]
        return matched_pred_ious, matched_gt_inds
