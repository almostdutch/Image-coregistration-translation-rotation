'''
translation_coreg_cross_corr.py

Script for performing corregestration (translation) of two 2D images by using cross correlation

CrossCorrDiamondSearchTranslation \
    Cross correlation (pixel level) with diamond search (sub-pixel level), image comparison metric = mean squared error

CrossCorrFullSearchTranslation \
    Cross correlation (pixel level) with full search (sub-pixel level), image comparison metric = mean squared error

CrossCorrDiffEvolutionTranslation \
    Cross correlation (pixel level) with differential evolution (sub-pixel level), image comparison metric = mean squared error
'''

import numpy as np
import scipy as scipy
from coreg_utils import ImageTranslate, _NormalizeImage, _ApplyHighPassFilter, _MseMetric, \
    _NeighborhoodDiamondSearch, _NeighborhoodFullSearch, _DiffEvolutionTranslation

def _CrossCorr(image_ref, image_shifted):
    # Determine pixel level shift
    
    Nrows, Ncols = image_ref.shape;
    image_shifted_180rot = np.flip(image_shifted, axis = (0, 1));
    cross_corr = scipy.signal.fftconvolve(image_ref, image_shifted_180rot, mode = 'same');

    ind = np.unravel_index(np.argmax(cross_corr, axis = None), cross_corr.shape)  
    shift_row = Nrows / 2 - ind[0];
    shift_col = Ncols / 2 - ind[1];
    return shift_row, shift_col;

def CrossCorrDiamondSearchTranslation(image_ref, image_shifted, Niter):
    image_ref = _NormalizeImage(image_ref);
    image_shifted = _NormalizeImage(image_shifted);
 
    image_ref = _ApplyHighPassFilter(image_ref);
    image_shifted = _ApplyHighPassFilter(image_shifted);
        
    # pixel level coregestration
    shift_row, shift_col = _CrossCorr(image_ref, image_shifted);    
    image_shifted = ImageTranslate(image_shifted, -shift_row, -shift_col);
    
    # sub-pixel level coregestration
    delta_row, delta_col = _NeighborhoodDiamondSearch(image_ref, image_shifted, 0, 0, Niter);
    return shift_row + delta_row, shift_col + delta_col;

def CrossCorrFullSearchTranslation(image_ref, image_shifted, Niter):
    image_ref = _NormalizeImage(image_ref);
    image_shifted = _NormalizeImage(image_shifted);
 
    image_ref = _ApplyHighPassFilter(image_ref);
    image_shifted = _ApplyHighPassFilter(image_shifted);
    
    # pixel level coregestration
    shift_row, shift_col = _CrossCorr(image_ref, image_shifted); 
    image_shifted = ImageTranslate(image_shifted, -shift_row, -shift_col); 
    
    # sub-pixel level coregestration
    delta_row, delta_col = _NeighborhoodFullSearch(image_ref, image_shifted, 0, 0, Niter);
    return shift_row + delta_row, shift_col + delta_col;

def CrossCorrDiffEvolutionTranslation(image_ref, image_shifted, bounds, Niter):
    image_ref = _NormalizeImage(image_ref);
    image_shifted = _NormalizeImage(image_shifted);
 
    image_ref = _ApplyHighPassFilter(image_ref);
    image_shifted = _ApplyHighPassFilter(image_shifted);
    
    # pixel level coregestration
    shift_row, shift_col = _CrossCorr(image_ref, image_shifted); 
    image_shifted = ImageTranslate(image_shifted, -shift_row, -shift_col); 
    
    # sub-pixel level coregestration
    obj_func = lambda shift: _MseMetric(image_ref, ImageTranslate(image_shifted, -shift[0], -shift[1]));       
    delta_row, delta_col = _DiffEvolutionTranslation(obj_func, bounds, Niter);     
    return shift_row + delta_row, shift_col + delta_col;