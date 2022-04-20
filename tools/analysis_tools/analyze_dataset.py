# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path
import warnings
from os import path as osp
# from pathlib import Path

import cv2
import mmcv
import numpy as np
from mmcv import Config, DictAction, mkdir_or_exist

# from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
#                                DepthInstance3DBoxes, LiDARInstance3DBoxes)
import matplotlib.pyplot as plt
import matplotlib.colors as clo


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['Normalize'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument(
        '--task',
        type=str,
        choices=['det', 'seg', 'multi_modality-det', 'mono-det'],
        help='Determine the visualization method depending on the task.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.output_dir is not None:
        mkdir_or_exist(args.output_dir)

    annos = mmcv.load('E:/public_datasets/kitti/kitti_infos_train_mono3d.coco.json')  # [:100]
    class_ana = []
    bev_pos_ana = []
    dim_ana = [[] for i in range(len(annos['categories']))]
    roty_ana = []
    bbox2d_ana = []
    focal_length_ana = []
    principal_point_offset_ana = []
    for anno in annos['annotations']:
        class_ana.append(anno['category_id'])
        bev_pos_ana.append(anno['bbox_cam3d'][0:3])
        dim_ana[anno['category_id']].append(anno['bbox_cam3d'][3:6])
        roty_ana.append(anno['bbox_cam3d'][6])
        bbox2d_ana.append(anno['bbox'])
    for image in annos['images']:
        focal_length_ana.append(image['cam_intrinsic'][0][0])
        principal_point_offset_ana.append((image['cam_intrinsic'][0][2],image['cam_intrinsic'][1][2]))
    fig = plt.figure()  # 创建画布
    ax = fig.subplots()  # 创建图表
    ax.hist(class_ana, bins=len(annos['categories']), ec='w', lw=1, color="C1")  # 设置分组数，设置edgecolor和linewidth
    plt.savefig(os.path.join(args.output_dir, 'class_dist.png'))

    bev_pos_ana = np.asarray(bev_pos_ana)
    fig = plt.figure()
    ax = fig.subplots()
    h = ax.hist2d(bev_pos_ana[:, 0], bev_pos_ana[:, 2], bins=50, cmap="Oranges")  # 设置分组数
    plt.colorbar(h[3], ax=ax)
    plt.savefig(os.path.join(args.output_dir, 'bev_dist.png'))

    fig = plt.figure()
    plt.figure(figsize=(30, 5))
    ax = plt.subplot(1, len(annos['categories']) + 1, 1)
    ax.set_title('all')
    h = ax.hist2d(np.concatenate(dim_ana, axis=0)[:, 0], np.concatenate(dim_ana, axis=0)[:, 2], bins=50,
                  cmap='Greys')  # 设置分组数
    plt.colorbar(h[3], ax=ax)
    for i in range(len(annos['categories'])):
        ax = plt.subplot(1, len(annos['categories']) + 1, i + 2)
        ax.set_title(annos['categories'][i])
        h = ax.hist2d(np.asarray(dim_ana[i])[:, 0], np.asarray(dim_ana[i])[:, 2], bins=10, cmap='Oranges')  # 设置分组数
        plt.colorbar(h[3], ax=ax)
    plt.savefig(os.path.join(args.output_dir, 'dim_dist.png'))

    fig = plt.figure()  # 创建画布
    ax = fig.subplots()  # 创建图表
    ax.hist(roty_ana, bins=24, ec='w', lw=1, color="C1")  # 设置分组数，设置edgecolor和linewidth
    plt.savefig(os.path.join(args.output_dir, 'roty_dist.png'))

    fig = plt.figure()  # 创建画布
    plt.subplots_adjust(hspace=0.5)  # 调整子图间距
    ax0 = plt.subplot(2, 1, 1)  # 创建图表
    ax0.set_title('bbox2d_sqrted_area')
    ax0.hist([np.sqrt(bbox2d[2] * bbox2d[3]) for bbox2d in bbox2d_ana], bins=100, ec='w', lw=1,
             color="C1")  # 设置分组数，设置edgecolor和linewidth
    ax1 = plt.subplot(2, 1, 2)  # 创建图表
    ax1.set_title('bbox2d_ratio')
    ax1.hist([bbox2d[2] / bbox2d[3] for bbox2d in bbox2d_ana], bins=24, ec='w', lw=1,
             color="C1")  # 设置分组数，设置edgecolor和linewidth
    plt.savefig(os.path.join(args.output_dir, 'bbox2d_dist.png'))

    fig = plt.figure()  # 创建画布
    ax = fig.subplots()  # 创建图表
    ax.hist(focal_length_ana, ec='w', lw=1, color="C1")  # 设置分组数，设置edgecolor和linewidth
    plt.savefig(os.path.join(args.output_dir, 'focal_length_dist.png'))

    principal_point_offset_ana = np.asarray(principal_point_offset_ana)
    fig = plt.figure()  # 创建画布
    ax = fig.subplots()  # 创建图表
    ax.hist(principal_point_offset_ana[:,0], ec='w', lw=1, color="C1")  # 设置分组数，设置edgecolor和linewidth
    plt.savefig(os.path.join(args.output_dir, 'principal_point_offset_dist.png'))

    return


if __name__ == '__main__':
    main()
