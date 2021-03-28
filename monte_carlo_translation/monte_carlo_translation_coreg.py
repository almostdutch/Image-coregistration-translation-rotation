'''
monte_carlo_translation_coreg.py

Script to perform monte carlo simulations comparing sub-pixel corregistration (translation) accuracy of seven methods:
    
    (1) Phase correlation (pixel level) with diamond search (sub-pixel level), image comparison metric = mean squared error
    (2) Phase correlation (pixel level) with full search (sub-pixel level), image comparison metric = mean squared error
    (3) Phase correlation (pixel level) with differential evolution (sub-pixel level), image comparison metric = mean squared error
    
    (4) Cross correlation (pixel level) with diamond search (sub-pixel level), image comparison metric = mean squared error
    (5) Cross correlation (pixel level) with full search (sub-pixel level), image comparison metric = mean squared error
    (6) Cross correlation (pixel level) with differential evolution (sub-pixel level), image comparison metric = mean squared error
    
    (7) Mutual information with differential evolution (sub-pixel level), image comparison metric = mutual information
'''

import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from coreg_utils import ImageTranslate
from translation_coreg_phase_corr import PhaseCorrDiamondSearchTranslation, \
    PhaseCorrFullSearchTranslation, PhaseCorrDiffEvolutionTranslation
from translation_coreg_cross_corr import CrossCorrDiamondSearchTranslation, \
    CrossCorrFullSearchTranslation, CrossCorrDiffEvolutionTranslation
from translation_coreg_mutual_info import MutualInfoDiffEvolutionTranslation
import random
import time

mu, sigma = 0, 4 # mean and standard deviation of added Gaussian noise
Niter = 50; # number of iterations
Ntrials = 2000; # number of monte carlo trials

# Preallocating arrays for book keeping
# 1st dimension for row errors, 2nd dimension for column errors
# Phase correlation methods
method1 = np.zeros((2, Ntrials)); 
method2 = np.zeros((2, Ntrials)); 
method3 = np.zeros((2, Ntrials)); 

# Cross correlation methods
method4 = np.zeros((2, Ntrials));
method5 = np.zeros((2, Ntrials));
method6 = np.zeros((2, Ntrials));

# Mutual information methods
method7 = np.zeros((2, Ntrials));

time_array = np.zeros((7, Ntrials));

# range of simulated pixel shifts
limit_left = -10.00;
limit_right = 10.00;

for i in range(Ntrials):
    print(i)
    image_ref = np.array(mpimg.imread('test_image.gif'));
    image_ref = image_ref.astype(np.float64);
    Nrows, Ncols = image_ref.shape;

    # generate random independent row and column pixel shift
    shift_row_rand = round(random.uniform(limit_left, limit_right), 2);
    shift_col_rand = round(random.uniform(limit_left, limit_right), 2);
    
    # print('Given shift:')
    # print('Row: ' + str(shift_row_rand) + ' Col: ' + str(shift_col_rand))
    
    image_shifted = ImageTranslate(image_ref.copy(), shift_row_rand, shift_col_rand);

    # add independent Gaussian noise for reference image (image_ref) and shifted image (image_shifted)
    image_ref += np.random.normal(mu, sigma, size = (Nrows, Ncols));
    image_shifted += np.random.normal(mu, sigma, size = (Nrows, Ncols));
    
    # Phase correlation methods    
    start = time.time();
    shift_row, shift_col = PhaseCorrDiamondSearchTranslation(image_ref.copy(), image_shifted.copy(), Niter);
    end = time.time();
    time_array[0][i] += (end - start);
    method1[0][i] = shift_row - shift_row_rand;
    method1[1][i] = shift_col - shift_col_rand;    
    
    start = time.time();
    shift_row, shift_col = PhaseCorrFullSearchTranslation(image_ref.copy(), image_shifted.copy(), Niter);
    end = time.time();
    time_array[1][i] += (end - start);    
    method2[0][i] = shift_row - shift_row_rand;
    method2[1][i] = shift_col - shift_col_rand;     

    start = time.time();
    bounds = [(-2, 2), (-2, 2)]; # bounds for sub-pixel level coregestration    
    shift_row, shift_col = PhaseCorrDiffEvolutionTranslation(image_ref.copy(), image_shifted.copy(), bounds, Niter);
    end = time.time();
    time_array[2][i] += (end - start);    
    method3[0][i] = shift_row - shift_row_rand;
    method3[1][i] = shift_col - shift_col_rand; 
    
    # Cross correlation methods
    start = time.time();
    shift_row, shift_col = CrossCorrDiamondSearchTranslation(image_ref.copy(), image_shifted.copy(), Niter);
    end = time.time();
    time_array[3][i] += (end - start);    
    method4[0][i] = shift_row - shift_row_rand;
    method4[1][i] = shift_col - shift_col_rand;    
    
    start = time.time();
    shift_row, shift_col = CrossCorrFullSearchTranslation(image_ref.copy(), image_shifted.copy(), Niter);
    end = time.time();
    time_array[4][i] += (end - start);        
    method5[0][i] = shift_row - shift_row_rand;
    method5[1][i] = shift_col - shift_col_rand;   

    start = time.time();
    bounds = [(-2, 2), (-2, 2)]; # bounds for sub-pixel level coregestration    
    shift_row, shift_col = CrossCorrDiffEvolutionTranslation(image_ref.copy(), image_shifted.copy(), bounds, Niter);
    end = time.time();
    time_array[5][i] += (end - start);    
    method6[0][i] = shift_row - shift_row_rand;
    method6[1][i] = shift_col - shift_col_rand;

    # Mutual information methods
    start = time.time();
    bounds = [(limit_left * 2, limit_right * 2), (limit_left * 2, limit_right * 2)]; # bounds for sub-pixel level coregestration    
    shift_row, shift_col = MutualInfoDiffEvolutionTranslation(image_ref.copy(), image_shifted.copy(), bounds, Niter);
    end = time.time();
    time_array[6][i] += (end - start);    
    method7[0][i] = shift_row - shift_row_rand;
    method7[1][i] = shift_col - shift_col_rand;
 
time_array_avg = np.mean(time_array, axis = 1);    
       
bins = np.linspace(-1, 1, 100);
weights = np.ones_like(method1[0]) / (len(method1[0]))
kfontsize = 8;

# Phase correlation Methods
fig_width, fig_height = 10, 5
plt.figure(figsize=(fig_width, fig_height))

plt.subplot(1,3,1)
plt.hist(method1[0], bins, color = '#0504aa', alpha = 0.5, label = 'rows', facecolor = 'blue', weights = weights)
plt.hist(method1[1], bins, color = '#0504aa', alpha = 0.5, label = 'columns', facecolor = 'red', weights = weights)
plt.legend(loc='upper right', prop={'size': 8})
plt.grid(axis='y', alpha = 0.75)
plt.xlabel('Error [pixels]')
plt.ylabel('Probability')
plt.title('diamond search')
plt.ylim((0, 0.2))
plt.xlim((-1, 1)) 
mu_row, sigma_row = np.mean(method1[0]), np.std(method1[0]);
mu_col, sigma_col = np.mean(method1[1]), np.std(method1[1]);
plt.text(-0.99,0.16, r'rows $ {} \pm {}$'.format(round(mu_row,2), round(sigma_row,2)), fontsize=kfontsize)
plt.text(-0.99,0.15, r'cols $ {} \pm {}$'.format(round(mu_col,2), round(sigma_col,2)), fontsize=kfontsize)
plt.text(-0.99,0.14, r'avg trial dur ${}ms$'.format(round(time_array_avg[0], 2), fontsize=kfontsize))
plt.subplots_adjust(hspace = .001)

plt.subplot(1,3,2)
plt.hist(method2[0], bins, color = '#0504aa', alpha = 0.5, label = 'rows', facecolor = 'blue', weights = weights)
plt.hist(method2[1], bins, color = '#0504aa', alpha = 0.5, label = 'columns', facecolor = 'red', weights = weights)
plt.legend(loc='upper right', prop={'size': 8})
plt.grid(axis='y', alpha = 0.75)
plt.xlabel('Error [pixels]')
plt.ylabel('Probability')
plt.title('full search')
plt.ylim((0, 0.2))
plt.xlim((-1, 1)) 
mu_row, sigma_row = np.mean(method2[0]), np.std(method2[0]);
mu_col, sigma_col = np.mean(method2[1]), np.std(method2[1]);
plt.text(-0.99,0.16, r'rows $ {} \pm {}$'.format(round(mu_row,2), round(sigma_row,2)), fontsize=kfontsize)
plt.text(-0.99,0.15, r'cols $ {} \pm {}$'.format(round(mu_col,2), round(sigma_col,2)), fontsize=kfontsize)
plt.text(-0.99,0.14, r'avg trial dur ${}ms$'.format(round(time_array_avg[1], 2), fontsize=kfontsize))
plt.subplots_adjust(hspace = .001)
plt.tight_layout()

plt.subplot(1,3,3)
plt.hist(method3[0], bins, color = '#0504aa', alpha = 0.5, label = 'rows', facecolor = 'blue', weights = weights)
plt.hist(method3[1], bins, color = '#0504aa', alpha = 0.5, label = 'columns', facecolor = 'red', weights = weights)
plt.legend(loc='upper right', prop={'size': 8})
plt.grid(axis='y', alpha = 0.75)
plt.xlabel('Error [pixels]')
plt.ylabel('Probability')
plt.title('differential evolution')
plt.ylim((0, 0.2))
plt.xlim((-1, 1)) 
mu_row, sigma_row = np.mean(method3[0]), np.std(method3[0]);
mu_col, sigma_col = np.mean(method3[1]), np.std(method3[1]);
plt.text(-0.99,0.16, r'rows $ {} \pm {}$'.format(round(mu_row,2), round(sigma_row,2)), fontsize=kfontsize)
plt.text(-0.99,0.15, r'cols $ {} \pm {}$'.format(round(mu_col,2), round(sigma_col,2)), fontsize=kfontsize)
plt.text(-0.99,0.14, r'avg trial dur ${}ms$'.format(round(time_array_avg[2], 2), fontsize=kfontsize))
plt.subplots_adjust(hspace = .001)
plt.tight_layout()

# Cross correlation methods
plt.figure(figsize=(fig_width, fig_height))
plt.subplot(1,3,1)
plt.hist(method4[0], bins, color = '#0504aa', alpha = 0.5, label = 'rows', facecolor = 'blue', weights = weights)
plt.hist(method4[1], bins, color = '#0504aa', alpha = 0.5, label = 'columns', facecolor = 'red', weights = weights)
plt.legend(loc='upper right', prop={'size': 8})
plt.grid(axis='y', alpha = 0.75)
plt.xlabel('Error [pixels]')
plt.ylabel('Probability')
plt.title('diamond search')
plt.ylim((0, 0.2))
plt.xlim((-1, 1)) 
mu_row, sigma_row = np.mean(method4[0]), np.std(method4[0]);
mu_col, sigma_col = np.mean(method4[1]), np.std(method4[1]);
plt.text(-0.99,0.16, r'rows $ {} \pm {}$'.format(round(mu_row,2), round(sigma_row,2)), fontsize=kfontsize)
plt.text(-0.99,0.15, r'cols $ {} \pm {}$'.format(round(mu_col,2), round(sigma_col,2)), fontsize=kfontsize)
plt.text(-0.99,0.14, r'avg trial dur ${}ms$'.format(round(time_array_avg[3], 2), fontsize=kfontsize))
plt.subplots_adjust(hspace = .001)

plt.subplot(1,3,2)
plt.hist(method5[0], bins, color = '#0504aa', alpha = 0.5, label = 'rows', facecolor = 'blue', weights = weights)
plt.hist(method5[1], bins, color = '#0504aa', alpha = 0.5, label = 'columns', facecolor = 'red', weights = weights)
plt.legend(loc='upper right', prop={'size': 8})
plt.grid(axis='y', alpha = 0.75)
plt.xlabel('Error [pixels]')
plt.ylabel('Probability')
plt.title('full search')
plt.ylim((0, 0.2))
plt.xlim((-1, 1)) 
mu_row, sigma_row = np.mean(method5[0]), np.std(method5[0]);
mu_col, sigma_col = np.mean(method5[1]), np.std(method5[1]);
plt.text(-0.99,0.16, r'rows $ {} \pm {}$'.format(round(mu_row,2), round(sigma_row,2)), fontsize=kfontsize)
plt.text(-0.99,0.15, r'cols $ {} \pm {}$'.format(round(mu_col,2), round(sigma_col,2)), fontsize=kfontsize)
plt.text(-0.99,0.14, r'avg trial dur ${}ms$'.format(round(time_array_avg[4], 2), fontsize=kfontsize))
plt.subplots_adjust(hspace = .001)

plt.subplot(1,3,3)
plt.hist(method6[0], bins, color = '#0504aa', alpha = 0.5, label = 'rows', facecolor = 'blue', weights = weights)
plt.hist(method6[1], bins, color = '#0504aa', alpha = 0.5, label = 'columns', facecolor = 'red', weights = weights)
plt.legend(loc='upper right', prop={'size': 8})
plt.grid(axis='y', alpha = 0.75)
plt.xlabel('Error [pixels]')
plt.ylabel('Probability')
plt.title('differential evolution')
plt.ylim((0, 0.2))
plt.xlim((-1, 1)) 
mu_row, sigma_row = np.mean(method6[0]), np.std(method6[0]);
mu_col, sigma_col = np.mean(method6[1]), np.std(method6[1]);
plt.text(-0.99,0.16, r'rows $ {} \pm {}$'.format(round(mu_row,2), round(sigma_row,2)), fontsize=kfontsize)
plt.text(-0.99,0.15, r'cols $ {} \pm {}$'.format(round(mu_col,2), round(sigma_col,2)), fontsize=kfontsize)
plt.text(-0.99,0.14, r'avg trial dur ${}ms$'.format(round(time_array_avg[5], 2), fontsize=kfontsize))
plt.subplots_adjust(hspace = .001)
plt.tight_layout()

# Mutual information methods
plt.figure(figsize=(fig_width, fig_height))
plt.subplot(1,3,1)
plt.hist(method7[0], bins, color = '#0504aa', alpha = 0.5, label = 'rows', facecolor = 'blue', weights = weights)
plt.hist(method7[1], bins, color = '#0504aa', alpha = 0.5, label = 'columns', facecolor = 'red', weights = weights)
plt.legend(loc='upper right', prop={'size': 8})
plt.grid(axis='y', alpha = 0.75)
plt.xlabel('Error [pixels]')
plt.ylabel('Probability')
plt.title('differential evolution')
plt.ylim((0, 0.2))
plt.xlim((-1, 1)) 
mu_row, sigma_row = np.mean(method7[0]), np.std(method7[0]);
mu_col, sigma_col = np.mean(method7[1]), np.std(method7[1]);
plt.text(-0.99,0.16, r'rows $ {} \pm {}$'.format(round(mu_row,2), round(sigma_row,2)), fontsize=kfontsize)
plt.text(-0.99,0.15, r'cols $ {} \pm {}$'.format(round(mu_col,2), round(sigma_col,2)), fontsize=kfontsize)
plt.text(-0.99,0.14, r'avg trial dur ${}ms$'.format(round(time_array_avg[6], 2), fontsize=kfontsize))
plt.subplots_adjust(hspace = .001)
plt.tight_layout()
