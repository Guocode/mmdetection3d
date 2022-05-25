# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.core.bbox import AssignResult, BaseAssigner, MaxIoUAssigner

from mmdet3d.core.bbox.assigners.dense_assigner_3d import DenseAssigner3D
from mmdet3d.core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D

__all__ = ['BaseAssigner', 'MaxIoUAssigner', 'AssignResult','HungarianAssigner3D','DenseAssigner3D','SimCTOTAAssigner','HungarianAssignerBEV']

from mmdet3d.core.bbox.assigners.hungarian_assigner_bev import HungarianAssignerBEV

from mmdet3d.core.bbox.assigners.sim_ct_ota_assigner import SimCTOTAAssigner

