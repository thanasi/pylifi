#############################
##
## MatchFeats.py
## Feature matching between images
##
## Athanasios Athanassiadis
## Sept. 2013
##
## MIT License
##
#############################
from __future__ import division
import numpy as np
import cv2

def get_dxdy(im1,im2, nkeep=100):
    """
    get the shift between two grayscale images

    assume they are taken from approximately the same distance
    so that we don't worry about scaling images

    nbest specifies the max number of keypoints to return

    """

    assert im1.ndim == im2.ndim, "get_dxdy() : please pass two images of the same dimensionality"

    if im1.ndim == 3:
        assert im1.shape[2] == 1, "get_dxdy() : please only input grayscale images"
        assert im2.shape[2] == 1, "get_dxdy() : please only input grayscale images"

    if im1.dtype is not np.dtype("np.uint8"):
        if im1.max() < 1:
            im1 *= 255
        im1 = im1.astype(np.uint8)
    if im2.dtype is not np.dtype("np.uint8"):
        if im2.max() < 1:
            im2 *= 255
        im2 = im2.astype(np.uint8)

    orb = cv2.ORB(nfeatures=nkeep)
    kp1,des1 = orb.detectAndCompute(im1,None)
    kp2,des2 = orb.detectAndCompute(im2,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key=lambda x:x.distance)

    dx = []
    dy = []
    da = []

    for match in matches:
        i1 = match.trainIdx
        i2 = match.queryIdx
        p1x,p1y = kp1[i1].pt
        p1a = kp1[i1].angle
        p2x,p2y = kp2[i2].pt
        p2a = kp2[i2].angle
        dx.append(p2x-p1x)
        dy.append(p2y-p1y)
        da.append(p2a-p1a)

    ddx = np.median(dx)
    ddy = np.median(dy)
    dda = np.median(da)

    return ddx,ddy
