# Copyright (c) OpenMMLab. All rights reserved.
import copy
import tempfile
from os import path as osp

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log

from mmdet.datasets import DATASETS

from . import KittiMonoDataset


@DATASETS.register_module()
class KittiMonoDepthDataset(KittiMonoDataset):

    def __init__(self,
                 data_root,
                 info_file,
                 load_interval=1,
                 with_velocity=False,
                 eval_version=None,
                 version=None,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            info_file=info_file,
            load_interval=load_interval,
            with_velocity=with_velocity,
            eval_version=eval_version,
            version=version,
            **kwargs)

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):

        if isinstance(results[0], dict):
            results_dict = dict()
            for k in results[0].keys():
                results_dict[k] = 0
            for result in results:
                for k,res in result.items():
                    results_dict[k]+=res
            for k in results_dict.keys():
                results_dict[k] = (results_dict[k]/len(results)).mean().item()

                print_log(
                    f'Results of {k}:\n' + str(results_dict[k]), logger=logger)

        return results_dict

