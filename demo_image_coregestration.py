'''
demo_image_coregestration.py

Coregister two 2D (single channel) same size images differing by translation and rotation
'''

import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from coreg_utils import ImageTranslate, ImageRotate
from translation_coreg_mutual_info import MutualInfoDiffEvolutionTranslation
from rotation_coreg_mutual_info import MutualInfoDiffEvolutionRotationCartesian
import random

# range for taking a random shift (pixels) and rotation angle (degrees)
limit_left = -10.00;
limit_right = 10.00;

mu, sigma = 0, 4 # mean and standard deviation of added Gaussian noise
Niter = 50; # number of iterations

# load test image
image_ref = np.array(mpimg.imread('test_image.gif'));
image_ref = image_ref.astype(np.float64);
Nrows, Ncols = image_ref.shape;

# generate random independent row and column pixel shift, and rotation angle
shift_row_rand = round(random.uniform(limit_left, limit_right), 2);
shift_col_rand = round(random.uniform(limit_left, limit_right), 2);
rot_angle_rand = round(random.uniform(limit_left, limit_right), 2);

# generated dummy image, shifted and rotated
image_shifted = ImageTranslate(image_ref.copy(), shift_row_rand, shift_col_rand);
image_rotated = ImageRotate(image_shifted.copy(), rot_angle_rand);

# add independent Gaussian noise for reference image (image_ref) and rotated image (image_rotated)
image_ref += np.random.normal(mu, sigma, size = (Nrows, Ncols));
image_rotated += np.random.normal(mu, sigma, size = (Nrows, Ncols));

# determine rotation angle 
bounds = [(limit_left*2, limit_right*2)]; # bounds for sub-pixel level coregestration    
rot_angle = MutualInfoDiffEvolutionRotationCartesian(image_ref.copy(), image_rotated.copy(), bounds, Niter);

# apply rotation correction
image_coreg = ImageRotate(image_rotated.copy(), -rot_angle);

# determine translation
bounds = [(limit_left*2, limit_right*2), (limit_left*2, limit_right*2)]; # bounds for sub-pixel level coregestration    
shift_row, shift_col = MutualInfoDiffEvolutionTranslation(image_ref.copy(), image_coreg.copy(), bounds, Niter);

# apply translation correction
image_coreg = ImageTranslate(image_coreg.copy(), -shift_row, -shift_col);

print('Expected rotation angle: ' + str(rot_angle_rand))
print('Determined rotation angle: ' + str(rot_angle))
print('Expected translation Row: ' + str(shift_row_rand) + ' Col: ' + str(shift_col_rand))
print('Determined translation Row: ' + str(shift_row) + ' Col: ' + str(shift_col))

fig_width, fig_height = 10, 5
plt.figure(figsize=(fig_width, fig_height))

# plot results
plt.subplot(1,3,1)
plt.imshow(image_ref)
plt.title('Ref Image')
plt.subplot(1,3,2)
plt.imshow(image_rotated)
plt.title('Dummy Image')
plt.subplot(1,3,3)
plt.imshow(image_coreg)
plt.title('Coreg Image')


