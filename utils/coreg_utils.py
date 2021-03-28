'''
coreg_utils.py

Utilities for performing corregistration (translation and rotation)
'''

import numpy as np
import scipy as scipy
import scipy.ndimage
import scipy.signal

def _CartesianToPolar(image_in):
    Nrows, Ncols = image_in.shape;
    xc, yc = Ncols / 2 - 0.5, Nrows / 2 - 0.5; # image center
    Rmax = int(min([xc, yc]));
    R = np.linspace(0, Rmax, 100); # polar radius
    Theta = np.linspace(0, 359, 360); # polar angle    
    image_polar = np.zeros((Theta.size, R.size));
    x = xc + np.cos(Theta[:,None] * 2 * np.pi / 360) * R[None,:];
    y = yc + np.sin(Theta[:,None] * 2 * np.pi / 360) * R[None,:];
    scipy.ndimage.map_coordinates(image_in, (y, x), order=1, output=image_polar)
    return image_polar;

def ImageTranslate(image_in, shift_row, shift_col):   
    image_out = scipy.ndimage.shift(image_in, (shift_row, shift_col));
    return image_out;

def ImageRotate(image_in, rot_angle):   
    image_out = scipy.ndimage.rotate(image_in, rot_angle, reshape=False);
    return image_out;

def _NormalizeImage(image_in):
    mean = np.mean(image_in, axis = (0, 1));
    top = image_in.copy() - mean;
    bottom = np.std(image_in, axis = (0, 1));
    image_out = top / bottom;
    return image_out;

def _ApplyHighPassFilter(image_in):   
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]);
    image_out = scipy.signal.convolve2d(image_in, kernel, boundary = 'fill', fillvalue = 0, mode = 'same');
    return image_out

def _ApplyHannWindow(image_in):  
    Nrows, Ncols = image_in.shape;
    hanning_window_2d = np.outer(np.hanning(Nrows).reshape(Nrows, 1), np.hanning(Ncols).reshape(1, Ncols));
    image_out = image_in * hanning_window_2d;
    return image_out

def _RemoveZeros(image_in1, image_in2):
    mask1 = image_in1 != 0;
    mask2 = image_in2 != 0;
    mask = mask1 * mask2;  
    image_out1 = image_in1[mask];
    image_out2 = image_in2[mask];
    return image_out1, image_out2

def _MseMetric(image_ref, image_shifted):
    # Calculate Mean squared error 
    
    image_ref, image_shifted = _RemoveZeros(image_ref, image_shifted);
    Nel = image_ref.size;  
    mse = 1 / Nel * np.sum(np.power(image_ref - image_shifted, 2));
    return mse

def _NeighborhoodDiamondSearch(image_ref, image_shifted, shift_row, shift_col, Niter):
    # Determine sub-pixel level shift based on neighborhood diamond search
    
    delta = 0.2; # initial guess
    epsilon = 10e-5; # tolerance
    row_array = np.zeros((1, Niter));
    col_array = np.zeros((1, Niter));
    
    for i in range(Niter):
        delta *= 0.9;
        metric_array = np.zeros((1, 5));
        metric_array[0][0] = _MseMetric(image_ref, ImageTranslate(image_shifted, -shift_row, -shift_col));
        metric_array[0][1] = _MseMetric(image_ref, ImageTranslate(image_shifted, -(shift_row + delta), -(shift_col)));
        metric_array[0][2] = _MseMetric(image_ref, ImageTranslate(image_shifted, -(shift_row - delta), -(shift_col)));
        metric_array[0][3] = _MseMetric(image_ref, ImageTranslate(image_shifted, -(shift_row), -(shift_col + delta)));
        metric_array[0][4] = _MseMetric(image_ref, ImageTranslate(image_shifted, -(shift_row), -(shift_col - delta)));
        ind = np.unravel_index(np.argmin(metric_array, axis = None), metric_array.shape);
        
        if i != 0:
            if np.abs(row_array[0][i] - row_array[0][i - 1]) < epsilon and np.abs(col_array[0][i] - col_array[0][i - 1]) < epsilon:
                break;      
        
        if ind[1] == 1:
            shift_row, shift_col = (shift_row + delta, shift_col);
        if ind[1] == 2:
            shift_row, shift_col = (shift_row - delta, shift_col);
        if ind[1] == 3:
            shift_row, shift_col = (shift_row, shift_col + delta);
        if ind[1] == 4:
            shift_row, shift_col = (shift_row, shift_col - delta);      

        row_array[0][1] = shift_row;
        col_array[0][1] = shift_col;
    return shift_row, shift_col;
        
def _NeighborhoodFullSearch(image_ref, image_shifted, shift_row, shift_col, Niter):
    # Determine sub-pixel level shift based on neighborhood full search

    delta = 0.2; # initial guess
    epsilon = 10e-5; # tolerance
    row_array = np.zeros((1, Niter));
    col_array = np.zeros((1, Niter));
    
    for i in range(Niter):
        delta *= 0.9;
        metric_array = np.zeros((1, 9));      
        metric_array[0][0] = _MseMetric(image_ref, ImageTranslate(image_shifted, -shift_row, -shift_col));
        metric_array[0][1] = _MseMetric(image_ref, ImageTranslate(image_shifted, -(shift_row + delta), -(shift_col)));
        metric_array[0][2] = _MseMetric(image_ref, ImageTranslate(image_shifted, -(shift_row - delta), -(shift_col)));
        metric_array[0][3] = _MseMetric(image_ref, ImageTranslate(image_shifted, -(shift_row), -(shift_col + delta)));
        metric_array[0][4] = _MseMetric(image_ref, ImageTranslate(image_shifted, -(shift_row), -(shift_col - delta)));
        metric_array[0][5] = _MseMetric(image_ref, ImageTranslate(image_shifted, -(shift_row + delta), -(shift_col + delta)));
        metric_array[0][6] = _MseMetric(image_ref, ImageTranslate(image_shifted, -(shift_row - delta), -(shift_col - delta)));
        metric_array[0][7] = _MseMetric(image_ref, ImageTranslate(image_shifted, -(shift_row + delta), -(shift_col - delta)));
        metric_array[0][8] = _MseMetric(image_ref, ImageTranslate(image_shifted, -(shift_row - delta), -(shift_col + delta)));            
        ind = np.unravel_index(np.argmin(metric_array, axis = None), metric_array.shape);

        if i != 0:
            if np.abs(row_array[0][i] - row_array[0][i - 1]) < epsilon and np.abs(col_array[0][i] - col_array[0][i - 1]) < epsilon:
                break;         
        
        if ind[1] == 1:
            shift_row, shift_col = (shift_row + delta, shift_col);
        if ind[1] == 2:
            shift_row, shift_col = (shift_row - delta, shift_col);
        if ind[1] == 3:
            shift_row, shift_col = (shift_row, shift_col + delta);
        if ind[1] == 4:
            shift_row, shift_col = (shift_row, shift_col - delta);            
        if ind[1] == 5:
            shift_row, shift_col = (shift_row + delta, shift_col + delta);
        if ind[1] == 6:
            shift_row, shift_col = (shift_row - delta, shift_col - delta);
        if ind[1] == 7:
            shift_row, shift_col = (shift_row + delta, shift_col - delta);
        if ind[1] == 8:
            shift_row, shift_col = (shift_row - delta, shift_col + delta);
            
        row_array[0][1] = shift_row;
        col_array[0][1] = shift_col;
    return shift_row, shift_col;
    
def _DiffEvolutionTranslation(obj_func, bounds, Niter):
    # Determine sub-pixel level shift based on differential evolution
          
    shift = scipy.optimize.differential_evolution(obj_func, bounds, maxiter = Niter);
    shift_row, shift_col = shift.x;
    return shift_row, shift_col;

def _DiffEvolutionRotation(obj_func, bounds, Niter):
    # Determine sub-degree level rotation based on differential evolution
          
    rot_angle = scipy.optimize.differential_evolution(obj_func, bounds, maxiter = Niter);
    rot_angle = rot_angle.x;
    return rot_angle[0];