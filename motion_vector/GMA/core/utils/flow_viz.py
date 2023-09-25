# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03

import numpy as np

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False, filter_upper_range=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi  # angle
    fk = (a+1) / 2*(ncols-1)  # color index based on angle (float)
    k0 = np.floor(fk).astype(np.int32)  # color index based on angle (int)
    k1 = k0 + 1
    k1[k1 == ncols] = 0  # corrected color index
    f = fk - k0  # float index - int index
    if filter_upper_range:
        rad[rad > 1] = 0  # out of range OF will show white color instead of col*0.75 (dark)
    for i in range(colorwheel.shape[1]):  # loop over R,G and B
        tmp = colorwheel[:,i]  # lookup table
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1  # interpolated color
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])  # col if rad==1; 1(white) if rad==0
        col[~idx] = col[~idx] * 0.75   # out of range # col*0.75 (deep color) if rad>1
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False, bbox=None, bboxes=None,
                  const_rad_max=None, filter_upper_range=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, -clip_flow, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]

    rad_max, flow_u, flow_v = flow_to_mean_uv(flow_uv, bboxes=bboxes, bbox=bbox, return_rad_max=True)
    epsilon = 1e-5
    if const_rad_max is not None:
        rad_max = const_rad_max
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr, filter_upper_range), flow_u, flow_v


def flow_to_mean_uv(flow_uv, bbox=None, bboxes=None, return_rad_max=False):
    rad_max = -1
    if bboxes is not None:
        n_box = len(bboxes)
        flow_v, flow_u = [-1]*(n_box-1), [-1]*(n_box-1)
        for i, bbx in enumerate(bboxes):
            if bbx is None:
                raise Exception('All of the bbox must be valid')
            flow_uv_cropped = flow_uv[bbx[2]:bbx[3], bbx[0]:bbx[1], :]
            u_cropped, v_cropped = flow_uv_cropped[:, :, 0], flow_uv_cropped[:, :, 1]
            if i < n_box-1:
                flow_v[i] = np.mean(v_cropped)
                flow_u[i] = np.mean(u_cropped)
            else:  # last bbox is to set the focus window
                if return_rad_max:
                    rad = np.sqrt(np.square(u_cropped) + np.square(v_cropped))
                    rad_max = np.percentile(rad, 85)
    elif bbox is not None:
        flow_uv_cropped = flow_uv[bbox[2]:bbox[3], bbox[0]:bbox[1], :]
        u_cropped, v_cropped = flow_uv_cropped[:, :, 0], flow_uv_cropped[:, :, 1]
        if return_rad_max:
            rad = np.sqrt(np.square(u_cropped) + np.square(v_cropped))
            rad_max = np.percentile(rad,85)
        flow_v = np.mean(v_cropped)
        flow_u = np.mean(v_cropped)
    else:
        u = flow_uv[:, :, 0]
        v = flow_uv[:, :, 1]
        if return_rad_max:
            rad = np.sqrt(np.square(u) + np.square(v))
            rad_max = np.max(rad)
        flow_v = np.mean(v)
        flow_u = np.mean(u)
    return rad_max, flow_u, flow_v


def img_to_mean_rgb(img, bbox=None, bboxes=None):  # img (RGB)

    if bboxes is not None:  # not used yet
        n_box = len(bboxes)
        r, g, b = [-1]*(n_box), [-1]*(n_box), [-1]*(n_box)
        for i, bbx in enumerate(bboxes):
            if bbx is None:
                raise Exception('All of the bbox must be valid')
            img_cropped = img[bbx[2]:bbx[3], bbx[0]:bbx[1], :]
            r_cropped, g_cropped, b_cropped = img_cropped[:, :, 0], \
                                              img_cropped[:, :, 1], img_cropped[:, :, 2]
            r[i] = np.mean(r_cropped)
            g[i] = np.mean(g_cropped)
            b[i] = np.mean(b_cropped)
        return r, g, b
    elif bbox is not None:
        img_cropped = img[bbox[2]:bbox[3], bbox[0]:bbox[1], :]
        r_cropped, g_cropped, b_cropped = img_cropped[:, :, 0], \
                                          img_cropped[:, :, 1], img_cropped[:, :, 2]
        r = np.mean(r_cropped)
        g = np.mean(g_cropped)
        b = np.mean(b_cropped)
        return [r, g, b]
    else:
        r_cropped, g_cropped, b_cropped = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        r = np.mean(r_cropped)
        g = np.mean(g_cropped)
        b = np.mean(b_cropped)
        return [r, g, b]

