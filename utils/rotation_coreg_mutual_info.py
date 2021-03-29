'''
rotation_coreg_mutual_info.py

Script for performing corregestration (rotation) of two 2D images by using mutual information \
    in cartesian coordinate system

MutualInfoDiffEvolutionRotationCartesian \
    Mutual information in Cartesian coordinate system with differential evolution (sub-degree level), \
    image comparison metric = mutual information
'''

import numpy as np
from coreg_utils import ImageRotate, _NormalizeImage, _ApplyHannWindow, _ApplyHighPassFilter, \
    _RemoveZeros, _DiffEvolutionRotation
    
def _MutualInfoMetric(image_ref, image_rotated):
    # Mutual information
    
    image_ref, image_rotated = _RemoveZeros(image_ref, image_rotated);
    hist_joint, _, _ = np.histogram2d(image_ref.flatten(), image_rotated.flatten(), bins = 256);
    hist_joint = hist_joint.T;
    p_xy = hist_joint / np.sum(hist_joint, axis = (0, 1));
    p_x = np.sum(p_xy, axis = 0);
    p_y = np.sum(p_xy, axis = 1);
    p_x_p_y = np.outer(p_y, p_x);
    mask = p_xy > 0;
    mutual_info = np.sum(p_xy[mask] * np.log(p_xy[mask] / p_x_p_y[mask]));
    return mutual_info;

def MutualInfoDiffEvolutionRotationCartesian(image_ref, image_rotated, bounds, Niter):
    image_ref = _NormalizeImage(image_ref);
    image_rotated = _NormalizeImage(image_rotated);
    
    image_ref = _ApplyHannWindow(image_ref);
    image_rotated = _ApplyHannWindow(image_rotated);
    
    image_ref = np.abs(np.fft.fftshift(np.fft.fft2(image_ref)));
    image_rotated = np.abs(np.fft.fftshift(np.fft.fft2(image_rotated)));
    
    image_ref = _ApplyHighPassFilter(image_ref);
    image_rotated = _ApplyHighPassFilter(image_rotated);
    
    # sub-degree level coregestration
    obj_func = lambda rot_angle: -_MutualInfoMetric(image_ref, ImageRotate(image_rotated, -rot_angle[0]));       
    rot_angle = _DiffEvolutionRotation(obj_func, bounds, Niter);   
    return rot_angle;