'''
monte_carlo_rotation_coreg.py

Script to perform monte carlo simulations comparing sub-degree coregistration (rotation) accuracy of 3 methods:
    
    (1) Phase correlation (degree level) in Polar coordinate system with differential evolution (sub-degree level), \
        image comparison metric = mean squared error
    
    (2) Cross correlation (degree level) in Polar coordinate system with differential evolution (sub-degree level), \
        image comparison metric = mean squared error
    
    (3) Mutual information in Cartesian coordinate system with differential evolution (sub-degree level), \
        image comparison metric = mutual information
'''

import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from coreg_utils import ImageRotate
from rotation_coreg_phase_corr import PhaseCorrDiffEvolutionRotationPolar
from rotation_coreg_cross_corr import CrossCorrDiffEvolutionRotationPolar
from rotation_coreg_mutual_info import MutualInfoDiffEvolutionRotationCartesian
import random
import time

mu, sigma = 0, 4 # mean and standard deviation of added Gaussian noise
Niter = 50; # number of iterations
Ntrials = 2000; # number of monte carlo trials

# Preallocating arrays for book keeping
# Phase correlation methods
method1 = np.zeros((1, Ntrials)); 

# Cross correlation methods
method2 = np.zeros((1, Ntrials));

# Mutual information methods
method3 = np.zeros((1, Ntrials));

time_array = np.zeros((3, Ntrials));

# range of simulated rotation angles
limit_left =  -10.00;
limit_right = 10.00;

for i in range(Ntrials):
    print(i)
    image_ref = np.array(mpimg.imread('test_image.gif'));
    image_ref = image_ref.astype(np.float64);
    Nrows, Ncols = image_ref.shape;

    # generate random rotation angle
    rot_angle_rand = round(random.uniform(limit_left, limit_right), 2);
    
    # print('Given rotation:')
    # print('Rot angle: ' + str(rot_angle_rand))
    
    image_rotated = ImageRotate(image_ref.copy(), rot_angle_rand);

    # add independent Gaussian noise for reference image (image_ref) and rotated image (image_rotated)
    image_ref += np.random.normal(mu, sigma, size = (Nrows, Ncols));
    image_rotated += np.random.normal(mu, sigma, size = (Nrows, Ncols));
    
    # Phase correlation methods    
    start = time.time();
    bounds = [(-1.5, 1.5)]; # bounds for sub-pixel level coregestration    
    rot_angle = PhaseCorrDiffEvolutionRotationPolar(image_ref.copy(), image_rotated.copy(), bounds, Niter);
    end = time.time();
    time_array[0][i] += (end - start);    
    method1[0][i] = rot_angle - rot_angle_rand;
    
    # Cross correlation methods
    start = time.time();
    bounds = [(-1.5, 1.5)]; # bounds for sub-pixel level coregestration    
    rot_angle = CrossCorrDiffEvolutionRotationPolar(image_ref.copy(), image_rotated.copy(), bounds, Niter);
    end = time.time();
    time_array[1][i] += (end - start);    
    method2[0][i] = rot_angle - rot_angle_rand;

    # Mutual information methods
    start = time.time();
    bounds = [(limit_left*2, limit_right*2)]; # bounds for sub-pixel level coregestration    
    rot_angle = MutualInfoDiffEvolutionRotationCartesian(image_ref.copy(), image_rotated.copy(), bounds, Niter);
    end = time.time();
    time_array[2][i] += (end - start);    
    method3[0][i] = rot_angle - rot_angle_rand;
 
time_array_avg = np.mean(time_array, axis = 1);    
       
bins = np.linspace(-1, 1, 100);
weights = np.ones_like(method1[0]) / (len(method1[0]))
kfontsize = 8;

fig_width, fig_height = 10, 5
plt.figure(figsize=(fig_width, fig_height))

# Phase correlation Methods
plt.subplot(1,3,1)
plt.hist(method1[0], bins, color = '#0504aa', alpha = 0.5, facecolor = 'blue', weights = weights)
plt.grid(axis='y', alpha = 0.75)
plt.xlabel('Error [degree]')
plt.ylabel('Probability')
plt.title('Phase corr')
plt.ylim((0, 0.2))
plt.xlim((-1, 1)) 
mu_row, sigma_row = np.mean(method1[0]), np.std(method1[0]);
plt.text(-0.99,0.16, r'rows $ {} \pm {}$'.format(round(mu_row,2), round(sigma_row,2)), fontsize=kfontsize)
plt.text(-0.99,0.15, r'avg trial dur ${}ms$'.format(round(time_array_avg[0], 2), fontsize=kfontsize))
plt.subplots_adjust(hspace = .001)

# Cross correlation methods
plt.subplot(1,3,2)
plt.hist(method2[0], bins, color = '#0504aa', alpha = 0.5, facecolor = 'blue', weights = weights)
plt.grid(axis='y', alpha = 0.75)
plt.xlabel('Error [degree]')
plt.ylabel('Probability')
plt.title('Cross corr')
plt.ylim((0, 0.2))
plt.xlim((-1, 1)) 
mu_row, sigma_row = np.mean(method2[0]), np.std(method2[0]);
plt.text(-0.99,0.16, r'rows $ {} \pm {}$'.format(round(mu_row,2), round(sigma_row,2)), fontsize=kfontsize)
plt.text(-0.99,0.15, r'avg trial dur ${}ms$'.format(round(time_array_avg[1], 2), fontsize=kfontsize))
plt.subplots_adjust(hspace = .001)

# Mutual information methods
plt.subplot(1,3,3)
plt.hist(method3[0], bins, color = '#0504aa', alpha = 0.5, facecolor = 'blue', weights = weights)
plt.grid(axis='y', alpha = 0.75)
plt.xlabel('Error [degree]')
plt.ylabel('Probability')
plt.title('Mutual info')
plt.ylim((0, 0.2))
plt.xlim((-1, 1)) 
mu_row, sigma_row = np.mean(method3[0]), np.std(method3[0]);
plt.text(-0.99,0.16, r'rows $ {} \pm {}$'.format(round(mu_row,2), round(sigma_row,2)), fontsize=kfontsize)
plt.text(-0.99,0.15, r'avg trial dur ${}ms$'.format(round(time_array_avg[2], 2), fontsize=kfontsize))
plt.subplots_adjust(hspace = .001)
plt.tight_layout()
