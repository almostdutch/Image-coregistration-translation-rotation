'''
translation_coreg_phase_corr.py

Script for performing corregestration (translation) of two 2D images by using phase correlation

Mutual information with differential evolution (sub-pixel level), image comparison metric = mutual information
'''

import numpy as np
from coreg_utils import ImageTranslate, _NormalizeImage, _ApplyHighPassFilter, _MseMetric, \
    _ApplyHannWindow, _NeighborhoodDiamondSearch, _NeighborhoodFullSearch, _DiffEvolutionTranslation

def _PhaseCorr(image_ref, image_shifted):
    # Determine pixel level shift
    
    Nrows, Ncols = image_ref.shape;    
    image_ref_fft = np.fft.fft2(image_ref);
    image_shifted_fft = np.fft.fft2(image_shifted);   
    top = np.multiply(np.matrix.conjugate(image_ref_fft), (image_shifted_fft));
    bottom = np.abs(top);
    phase_corr = np.real(np.fft.ifft2((np.divide(top, bottom))));

    ind = np.unravel_index(np.argmax(phase_corr), phase_corr.shape) 
    if ind[0] > Nrows/2:
        shift_row = ind[0] - Nrows; # to correctly handle negative shift
    else:
        shift_row = ind[0]
        
    if ind[1] > Ncols/2:
        shift_col = ind[1] - Ncols; # to correctly handle negative shift
    else:
        shift_col = ind[1]
    return phase_corr, shift_row, shift_col;

def PhaseCorrDiamondSearchTranslation(image_ref, image_shifted, Niter):
    image_ref = _NormalizeImage(image_ref);
    image_shifted = _NormalizeImage(image_shifted);
 
    image_ref = _ApplyHighPassFilter(image_ref);
    image_shifted = _ApplyHighPassFilter(image_shifted);

    image_ref = _ApplyHannWindow(image_ref);
    image_shifted = _ApplyHannWindow(image_shifted);
    
    # pixel level coregestration
    _, shift_row, shift_col = _PhaseCorr(image_ref, image_shifted); 
    image_shifted = ImageTranslate(image_shifted, -shift_row, -shift_col); 
    
    # sub-pixel level coregestration
    delta_row, delta_col = _NeighborhoodDiamondSearch(image_ref, image_shifted, 0, 0, Niter);       
    return shift_row + delta_row, shift_col + delta_col;

def PhaseCorrFullSearchTranslation(image_ref, image_shifted, Niter):
    image_ref = _NormalizeImage(image_ref);
    image_shifted = _NormalizeImage(image_shifted);
 
    image_ref = _ApplyHighPassFilter(image_ref);
    image_shifted = _ApplyHighPassFilter(image_shifted);

    image_ref = _ApplyHannWindow(image_ref);
    image_shifted = _ApplyHannWindow(image_shifted);
    
    # pixel level coregestration
    _, shift_row, shift_col = _PhaseCorr(image_ref, image_shifted); 
    image_shifted = ImageTranslate(image_shifted, -shift_row, -shift_col);
    
    # sub-pixel level coregestration
    delta_row, delta_col = _NeighborhoodFullSearch(image_ref, image_shifted, 0, 0, Niter);        
    return shift_row + delta_row, shift_col + delta_col;

def PhaseCorrDiffEvolutionTranslation(image_ref, image_shifted, bounds, Niter):
    image_ref = _NormalizeImage(image_ref);
    image_shifted = _NormalizeImage(image_shifted);
 
    image_ref = _ApplyHighPassFilter(image_ref);
    image_shifted = _ApplyHighPassFilter(image_shifted);

    image_ref = _ApplyHannWindow(image_ref);
    image_shifted = _ApplyHannWindow(image_shifted);
    
    # pixel level coregestration
    _, shift_row, shift_col = _PhaseCorr(image_ref, image_shifted); 
    image_shifted = ImageTranslate(image_shifted, -shift_row, -shift_col);
    
    # sub-pixel level coregestration
    obj_func = lambda shift: _MseMetric(image_ref, ImageTranslate(image_shifted, -shift[0], -shift[1]));       
    delta_row, delta_col = _DiffEvolutionTranslation(obj_func, bounds, Niter);       
    return shift_row + delta_row, shift_col + delta_col;
