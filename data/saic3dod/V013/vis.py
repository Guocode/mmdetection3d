import yaml
from yaml import load, dump
import numpy as np
from yaml import CUnsafeLoader as Loader
# from yaml import Loader, Dumper
import cv2
from pyquaternion import Quaternion

import open3d as o3d

import numpy as np

pcd = o3d.io.read_point_cloud('./V013_202203081800/choose_upload/2022-03-08-18-00/pcd/1646733652542173000.pcd')
points = np.array(pcd.points)  # 转为矩阵
valid_points = points[np.logical_not(np.isnan(points).any(-1))]
valid_points_homo = np.concatenate([valid_points,np.ones((valid_points.shape[0],1))],axis=-1)
K = np.asarray([[3149.167165, 0., 1915.019535, ],
                [0., 3149.167165, 1080.251082, ],
                [0., 0., 1.]])
R = np.asarray([4.9907094685009518e-01, -4.9554179034423818e-01,
       -4.9853557745834781e-01, 5.0678277015686035e-01 ])
t = np.asarray([-1.7684924597337990e+00, -3.3845262249688549e-02, -5.1838374389814934e-01])
r_quat = Quaternion(R)
r_mat = r_quat.rotation_matrix
rt_pcd = r_mat@valid_points.T -t[...,None]

# rt_pcd = rt_pcd[[1,2,0],] # zxy   xyz
rt_pcd = np.stack([-rt_pcd[2],rt_pcd[0],-rt_pcd[1]],axis=0)

# valid_rt_pcd = rt_pcd[:,rt_pcd[0,:]>0]
velo_pts_im = (K@(rt_pcd)).T
velo_pts_im[:,:2] = velo_pts_im[:,:2]/velo_pts_im[:,2:]
# velo_pts_im = (velo_pts_im[:2]/velo_pts_im[2:]).T
img = cv2.imread('./V013_202203081800/choose_upload/2022-03-08-18-00/jpg/1646733652542173000.jpg')
im_shape = img.shape
# check if in bounds
# use minus 1 to get the exact same value as KITTI matlab code
velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
velo_pts_im = velo_pts_im[val_inds, :]
print(velo_pts_im.shape)
# project to image
depth = np.zeros((im_shape[:2]), dtype=np.float32)
depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

img = cv2.resize(img, (0, 0), fx=0.15, fy=0.15)
depth = cv2.resize(depth, (0, 0), fx=0.15, fy=0.15)

cv2.imshow('a', img)
cv2.imshow('b', depth)

cv2.waitKey(0)
