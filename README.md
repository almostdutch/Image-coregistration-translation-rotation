# Image-coregistration-translation-rotation<br/>
Python implementation of Fourier-Mellin transform to coregister (translation with sub-pixel and rotation with sub-degree accuracy) two 2D (single channel) images 

### monte_carlo_translation_coreg.py<br/>
The script performs Monte Carlo simulations (Ntrials = 2000, -10 pixels <= shift <= 10 pixels) comparing sub-pixel corregistration (translation) accuracy of seven methods:
    
(1) Phase correlation (pixel level) with diamond search (sub-pixel level), image comparison metric = mean squared error<br/>
(2) Phase correlation (pixel level) with full search (sub-pixel level), image comparison metric = mean squared error<br/>
(3) Phase correlation (pixel level) with differential evolution (sub-pixel level), image comparison metric = mean squared error<br/>
    
<p align="center">
  <img src="monte_carlo_translation/translation_phase_corr.png" width="620" height="240"/>
</p>
    
(4) Cross correlation (pixel level) with diamond search (sub-pixel level), image comparison metric = mean squared error<br/>
(5) Cross correlation (pixel level) with full search (sub-pixel level), image comparison metric = mean squared error<br/>
(6) Cross correlation (pixel level) with differential evolution (sub-pixel level), image comparison metric = mean squared error<br/>

<p align="center">
  <img src="monte_carlo_translation/translation_cross_corr.png" width="620" height="240"/>
</p>

(7) Mutual information with differential evolution (sub-pixel level), image comparison metric = mutual information


<p align="center">
  <img src="monte_carlo_translation/translation_mutual_info.png" width="260" height="240"/>
</p>

### monte_carlo_rotation_coreg.py<br/>
The script performs Monte Carlo simulations (Ntrials = 2000, -10 pixels <= shift <= 10 degrees) comparing sub-degree coregistration (rotation) accuracy of 3 methods:
    
(1) Phase correlation (degree level) in Polar coordinate system with differential evolution (sub-degree level),<br/>
image comparison metric = mean squared error<br/>
(2) Cross correlation (degree level) in Polar coordinate system with differential evolution (sub-degree level),<br/>
image comparison metric = mean squared error<br/>
(3) Mutual information in Cartesian coordinate system with differential evolution (sub-degree level),<br/>
image comparison metric = mutual information<br/>
    
<p align="center">
  <img src="monte_carlo_rotation/rotation_phasecorr_crosscorr_mutualinfo.png" width="620" height="240"/>
</p>
    
### demo_image_coregestration.py<br/>
Demo showing how to coregister two 2D images

<p align="center">
  <img src="demo_results.png" width="620" height="240"/>
</p>
    
Expected rotation angle: 7.96<br/>
Determined rotation angle: 7.9863800832673615<br/>
Expected translation Row: -0.42 Col: -9.64<br/>
Determined translation Row: -0.4357482894087572 Col: -9.65762680684138<br/>
