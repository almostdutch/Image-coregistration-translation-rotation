'''
rotation_coreg_phase_corr.py

Script for performing corregistration (rotation) of two 2D images by using phase correlation \
    in polar coordinate system

PhaseCorrDiffEvolutionRotationPolar \
    Phase correlation (degree level) in Polar coordinate system with differential evolution (sub-degree level), \
    image comparison metric = mean squared error
'''

import numpy as np
from utils.coreg_utils import ImageTranslate, _NormalizeImage, _ApplyHighPassFilter, _ApplyHannWindow, \
    _MseMetric, _CartesianToPolar, _DiffEvolutionRotation

def _PhaseCorr(image_ref, image_shifted):
    # Determine pixel level shift
    
    Nrows, Ncols = image_ref.shape;    
    image_ref_fft = np.fft.fft2(image_ref);
    image_shifted_fft = np.fft.fft2(image_shifted);   
    top = np.multiply(np.matrix.conjugate(image_ref_fft), (image_shifted_fft));
    bottom = np.abs(top);
    phase_corr = np.real(np.fft.ifft2((np.divide(top, bottom))));

    ind = np.unravel_index(np.argmax(phase_corr, axis = None), phase_corr.shape) 
    if ind[0] > Nrows/2:
        shift_row = ind[0] - Nrows; # to correctly handle negative shift
    else:
        shift_row = ind[0]
        
    if ind[1] > Ncols/2:
        shift_col = ind[1] - Ncols; # to correctly handle negative shift
    else:
        shift_col = ind[1]
    return shift_row, shift_col;

def PhaseCorrDiffEvolutionRotationPolar(image_ref, image_rotated, bounds, Niter):
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
    shift_row, _ = _PhaseCorr(image_ref, image_rotated);    
    image_rotated = ImageTranslate(image_rotated, -shift_row, 0); 

    # sub-pixel level coregestration
    obj_func = lambda shift: _MseMetric(image_ref, ImageTranslate(image_rotated, -shift[0], 0));       
    delta_row = _DiffEvolutionRotation(obj_func, bounds, Niter); 
    rot_angle = -(shift_row + delta_row);   
    return rot_angle;

