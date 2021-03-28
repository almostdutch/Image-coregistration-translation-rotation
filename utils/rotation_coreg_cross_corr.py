'''
rotation_coreg_cross_corr.py

Script for performing corregestration (rotation) of two 2D images by using cross correlation \
    in polar coordinate system

- Cross correlation (degree level) in Polar coordinate system with differential evolution (sub-degree level), \
    image comparison metric = mean squared error
'''

import numpy as np
import scipy as scipy
from coreg_utils import ImageTranslate, _NormalizeImage, _ApplyHighPassFilter, _ApplyHannWindow, \
    _MseMetric, _CartesianToPolar, _DiffEvolutionRotation

def _CrossCorr(image_ref, image_shifted):
    # Determine pixel level shift
    
    Nrows, Ncols = image_ref.shape;
    image_shifted_180rot = np.flip(image_shifted, axis = (0, 1));
    cross_corr = scipy.signal.fftconvolve(image_ref, image_shifted_180rot, mode = 'same');

    ind = np.unravel_index(np.argmax(cross_corr, axis = None), cross_corr.shape)  
    shift_row = Nrows / 2 - ind[0];
    shift_col = Ncols / 2 - ind[1];    
    return shift_row, shift_col;

def CrossCorrDiffEvolutionRotationPolar(image_ref, image_rotated, bounds, Niter):
    image_ref = _NormalizeImage(image_ref);
    image_rotated = _NormalizeImage(image_rotated);

    image_ref = _ApplyHannWindow(image_ref);
    image_rotated = _ApplyHannWindow(image_rotated);
    
    image_ref = np.abs(np.fft.fftshift(np.fft.fft2(image_ref)));
    image_rotated = np.abs(np.fft.fftshift(np.fft.fft2(image_rotated)));

    image_ref = _ApplyHighPassFilter(image_ref);
    image_rotated = _ApplyHighPassFilter(image_rotated);
    
    image_ref = _CartesianToPolar(image_ref);
    image_rotated = _CartesianToPolar(image_rotated);
    
    # pixel level coregestration
    shift_row, _ = _CrossCorr(image_ref, image_rotated);    
    shift_row = shift_row / 1;
    image_rotated = ImageTranslate(image_rotated, -shift_row, 0); 
    
    # sub-pixel level coregestration
    obj_func = lambda shift: _MseMetric(image_ref, ImageTranslate(image_rotated, -shift[0], 0));       
    delta_row = _DiffEvolutionRotation(obj_func, bounds, Niter); 
    rot_angle = -(shift_row + delta_row);   
    return rot_angle;

