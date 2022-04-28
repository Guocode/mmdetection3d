# Copyright (c) OpenMMLab. All rights reserved.
import os.path
from collections import OrderedDict
from pathlib import Path
import mmcv
import numpy as np

from collections import Counter


def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points


def generate_depth_map(cam2cam, velo2cam, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    """
    # load calibration files
    # cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    # velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))

    # velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    # velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
    velo2cam = velo2cam['RT']
    # get image shape
    im_shape = cam2cam["S_rect_02"][::].astype(np.int32)

    # compute projection matrix velodyne->image plane
    # R_cam2rect = np.eye(4)
    R_cam2rect = cam2cam['R_rect_00']  # .reshape(3, 3)
    P_rect = cam2cam['P_rect_0' + str(cam)][:3, :4].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]), dtype=np.float32)
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth = np.clip(depth, a_min=0, a_max=200)
    # (0,500)->(0,65535)
    depth = (np.power(depth, 1 / 2) / np.power(200., 1 / 2) * 65535.).astype(np.uint16)
    return depth

def gen_depth_add_path(rootdir,info_pkl,depth_dir):
    infos = mmcv.load(os.path.join(rootdir,info_pkl))
    for info in mmcv.track_iter_progress(infos):
        imgfp = os.path.join(rootdir, info['image']['image_path'])
        cam2cam = {'S_rect_02': info['image']['image_shape'], 'R_rect_00': info['calib']['R0_rect'],
                   'P_rect_02': info['calib']['P2']}
        velo2cam = {'RT': info['calib']['Tr_velo_to_cam']}
        velo_filename = os.path.join(rootdir, info['point_cloud']['velodyne_path'])
        depth = generate_depth_map(cam2cam, velo2cam, velo_filename)
        # mmcv.imshow(imgfp)
        # mmcv.imshow(depth)
        mmcv.imwrite(depth, os.path.join(rootdir, info['image']['image_path'].replace('image_2', depth_dir)))
        info['image']['depth_path'] = info['image']['image_path'].replace('image_2', depth_dir)
    mmcv.dump(infos,os.path.join(rootdir,info_pkl))
if __name__ == '__main__':
    gen_depth_add_path('E:/public_datasets/mini_kitti','kitti_infos_train.pkl','depth')
    gen_depth_add_path('E:/public_datasets/mini_kitti','kitti_infos_val.pkl','depth')