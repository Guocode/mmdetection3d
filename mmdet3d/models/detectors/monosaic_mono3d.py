import numpy as np
import torch

from mmdet3d.core import (CameraInstance3DBoxes, bbox3d2result,
                          mono_cam_box2vis, show_multi_modality_result)

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector


@DETECTORS.register_module()
class MonoSAIC(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        assert bbox_head is not None
        super(MonoSAIC, self).__init__(backbone, neck, bbox_head, train_cfg,
                                       test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_3d,
                      kpts2d,
                      kpts2d_valid,
                      attr_labels=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_3d,
                                              kpts2d, kpts2d_valid,
                                              attr_labels, gt_bboxes_ignore,
                                              **kwargs)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_outputs = self.bbox_head.get_bboxes(*outs, img_metas)
        from mmdet.core import bbox2result
        bbox2d_img = []
        bbox_img = []
        for bbox_output in bbox_outputs:
            bboxes, bboxes3d,labels = bbox_output
            bbox2d_img.append(bbox2result(bboxes, labels, self.bbox_head.num_classes))
            bbox_img.append(bbox3d2result(bboxes3d, bboxes[..., -1], labels))

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox
        for result_dict, img_bbox2d in zip(bbox_list, bbox2d_img):
            result_dict['img_bbox2d'] = img_bbox2d
        return bbox_list

    def simple_depth_test(self, img, img_metas, densedepth=None, rescale=False):
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, densedepth=densedepth[0])
        return (losses,)

    def aug_test(self, imgs, img_metas, rescale=True):
        raise NotImplementedError

    def show_results(self, data, result, out_dir):
        raise NotImplementedError
