'''
translation_coreg_mutual_info.py

Script for performing corregestration (translation) of two 2D images by using mutual information

MutualInfoDiffEvolutionTranslation \
    Mutual information with differential evolution (sub-pixel level), image comparison metric = mutual information
'''

import numpy as np
from utils.coreg_utils import ImageTranslate, _NormalizeImage, _ApplyHighPassFilter, \
    _RemoveZeros, _DiffEvolutionTranslation
    
def _MutualInfoMetric(image_ref, image_shifted):
    # Mutual information

    image_ref, image_shifted = _RemoveZeros(image_ref, image_shifted);    
    hist_joint, _, _ = np.histogram2d(image_ref, image_shifted, bins = 256);
    hist_joint = hist_joint.T;
    p_xy = hist_joint / np.sum(hist_joint, axis = (0, 1));
    p_x = np.sum(p_xy, axis = 0);
    p_y = np.sum(p_xy, axis = 1);
    p_x_p_y = np.outer(p_y, p_x);
    mask = p_xy > 0;
    mutual_info = np.sum(p_xy[mask] * np.log(p_xy[mask] / p_x_p_y[mask]));
    return mutual_info;

def MutualInfoDiffEvolutionTranslation(image_ref, image_shifted, bounds, Niter):
    image_ref = _NormalizeImage(image_ref);
    image_shifted = _NormalizeImage(image_shifted);
 
    image_ref = _ApplyHighPassFilter(image_ref);
    image_shifted = _ApplyHighPassFilter(image_shifted);
    
    # sub-pixel level coregestration
    obj_func = lambda shift: -_MutualInfoMetric(image_ref, ImageTranslate(image_shifted, -shift[0], -shift[1]));       
    shift_row, shift_col = _DiffEvolutionTranslation(obj_func, bounds, Niter);         
    return shift_row, shift_col;