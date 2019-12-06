# Goal
The goal of this repo is to pull all plateifu ids from MaNGA and run them through a code that extracts the RA, DEC, and SDSS information such as camcol, run, etc, in order to process the SDSS images. GalaxySmelter will then measure all the imaging values and MergerMonger will utilize the LDA to classify galaxies as merging or nonmerging.

# Data
My goal is to share publicly released MaNGA data here, so I will need to make sure that the dap version lines up with the most recent public data release (currently I am using dapall-v2_5_3-2.3.0.fits).

# Classification
The classification is currently drawn at p = 0.5, and I am currently investigating the possibility of creating various different classification thresholds to make cleaner samples.
<img src="Panel_7815-6101.png"> 

# Of additional note
For MaNGA folks who would like to go ahead and start using these tools - please contact me for a chat about the limitations of the method and please refer to <a href="https://arxiv.org/abs/1901.01975">Nevin et al. 2019</a> for details. The main limitations are with <S/N> limitations (not to be used for galaxies with an average S/N per pixel of 2.5) and with redshift (be very cautious using this for galaxies with z>0.5). This last point is both due to surface brightness and angular resolution limitations but is also related to the inherent different structure of higher redshift galaxies (we would want to recreate the training sample for higher z galaxies to do this well). I am also currently investigating limitations with bulge to total mass ratio. This technique has been trained on diskier galaxies so I am ensuring that it functions well for various morphological types in SDSS and MaNGA. 
