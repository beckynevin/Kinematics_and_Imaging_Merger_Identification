'''

GalaxySmelter - This is the 'Real' edition to be applied to a real galaxy survey to extract predictors and place them in a table.

Things to install: Source Extractor, Galfit, statmorph (cite Vicente Rodriguez-Gomez)

The first step is to determine how to obtain galaxy images, ivar images, and psf; here I show an example for MaNGA preimaging,
utilizing wget - you can make your own directory that contains files, ivar files, and psfs, just name them camera_data, camera_data_ivar,
and psf, respectively.

These images need to be in units of counts.

Here, I run everything from within a folder that has subfolders imaging/ and preim/
which contain the output check images and the input images, respectively.
'''

'''
Import things, these first few lines are so that this version of python plays nicely with plotting on the supercomputer
'''
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()    

from scipy.ndimage import rotate
from skimage import measure
import numpy as np
from astropy.io import fits
import os
import matplotlib
from astropy.cosmology import WMAP9 as cosmo
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
import photutils
from scipy import ndimage
import statmorph #credit to Vicente Rodriguez-Gomez ()
import scipy
import pyfits
import math
import numpy.ma as ma
import pandas as pd

def SDSS_objid_to_values(objid):

    # Determined from http://skyserver.sdss.org/dr7/en/help/docs/algorithm.asp?key=objID

    bin_objid = bin(objid)
    bin_objid = bin_objid[2:len(bin_objid)]
    bin_objid = bin_objid.zfill(64)

    empty = int( '0b' + bin_objid[0], base=0)
    skyVersion = int( '0b' + bin_objid[1:4+1], base=0)
    rerun = int( '0b' + bin_objid[5:15+1], base=0)
    run = int( '0b' + bin_objid[16:31+1], base=0)
    camcol = int( '0b' + bin_objid[32:34+1], base=0)
    firstField = int( '0b' + bin_objid[35+1], base=0)
    field = int( '0b' + bin_objid[36:47+1], base=0)
    object_num = int( '0b' + bin_objid[48:63+1], base=0)

    return skyVersion, rerun, run, camcol, field, object_num



'''Prepares the default file for source extractor to read (I played with the parameters to the point where they work well:'''
def write_sex_default(name):
    file = open('imaging/default_'+name+'.sex', "w")
    file.write('# Default configuration file for SExtractor 2.5.0'+'\n')
        
    file.write('CATALOG_NAME     imaging/test_'+name+'.cat       # name of the output catalog'+'\n')
    file.write('CATALOG_TYPE     ASCII_HEAD     # '+'\n')
    file.write('PARAMETERS_NAME  imaging/default.param  # name of the file containing catalog contents'+'\n')

    file.write('#------------------------------- Extraction ----------------------------------'+'\n')

    file.write('DETECT_TYPE      CCD            # CCD (linear) or PHOTO (with gamma correction)'+'\n')
    file.write('DETECT_MINAREA   2            # minimum number of pixels above threshold'+'\n')
    file.write('DETECT_THRESH    1.5            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2'+'\n')
    file.write('ANALYSIS_THRESH  2            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2'+'\n')

    file.write('FILTER           Y              # apply filter for detection (Y or N)?'+'\n')
    file.write('FILTER_NAME      imaging/default.conv   # name of the file containing the filter'+'\n')

    file.write('DEBLEND_NTHRESH  32             # Number of deblending sub-thresholds'+'\n')
    file.write('DEBLEND_MINCONT  0.005          # Minimum contrast parameter for deblending'+'\n')

    file.write('CLEAN            Y              # Clean spurious detections? (Y or N)?'+'\n')
    file.write('CLEAN_PARAM      1.0            # Cleaning efficiency'+'\n')

    file.write('MASK_TYPE        CORRECT        # type of detection MASKing: can be one of'+'\n')
                                # NONE, BLANK or CORRECT

    file.write('WEIGHT_TYPE      NONE'+'\n')

    file.write('#------------------------------ Photometry -----------------------------------'+'\n')

    file.write('PHOT_APERTURES   5              # MAG_APER aperture diameter(s) in pixels'+'\n')
    file.write('PHOT_AUTOPARAMS  2.5, 3.5       # MAG_AUTO parameters: <Kron_fact>,<min_radius>'+'\n')
    file.write('PHOT_PETROPARAMS 2.0, 3.5       # MAG_PETRO parameters: <Petrosian_fact>,'+'\n')
                                # <min_radius>

    file.write('SATUR_LEVEL      50000.0        # level (in ADUs) at which arises saturation'+'\n')

    file.write('MAG_ZEROPOINT    0.0            # magnitude zero-point'+'\n')
    file.write('MAG_GAMMA        4.0            # gamma of emulsion (for photographic scans)'+'\n')
    file.write('GAIN             0.0            # detector gain in e-/ADU'+'\n')
    file.write('PIXEL_SCALE      1.0            # size of pixel in arcsec (0=use FITS WCS info)'+'\n')

    file.write('#------------------------- Star/Galaxy Separation ----------------------------'+'\n')

    file.write('SEEING_FWHM      1.2            # stellar FWHM in arcsec'+'\n')
    file.write('STARNNW_NAME     imaging/default.nnw    # Neural-Network_Weight table filename'+'\n')

    file.write('#------------------------------ Background -----------------------------------'+'\n')

    file.write('BACK_SIZE        30             # Background mesh: <size> or <width>,<height>'+'\n')
    file.write('BACK_FILTERSIZE  3              # Background filter: <size> or <width>,<height>'+'\n')
    file.write('BACKPHOTO_TYPE   LOCAL         # can be GLOBAL or LOCAL'+'\n')

    file.write('#------------------------------ Check Image ----------------------------------'+'\n')

    file.write('CHECKIMAGE_TYPE APERTURES         # can be NONE, BACKGROUND, BACKGROUND_RMS,'+'\n')
                                # MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND,
                                # FILTERED, OBJECTS, -OBJECTS, SEGMENTATION,
                                # or APERTURES
    file.write('CHECKIMAGE_NAME  imaging/aps_'+name+'.fits     # Filename for the check-image'+'\n')

    file.write('#--------------------- Memory (change with caution!) -------------------------'+'\n')

    file.write('MEMORY_OBJSTACK  3000           # number of objects in stack'+'\n')
    file.write('MEMORY_PIXSTACK  300000         # number of pixels in stack'+'\n')
    file.write('MEMORY_BUFSIZE   1024           # number of lines in buffer'+'\n')

    file.write('#----------------------------- Miscellaneous ---------------------------------'+'\n')

    file.write('VERBOSE_TYPE     QUIET        # can be QUIET, NORMAL or FULL'+'\n')
    file.write('WRITE_XML        Y              # Write XML file (Y/N)?'+'\n')
    file.write('XML_NAME         imaging/sex.xml        # Filename for XML output'+'\n')


    file.close()

'''Runs source extractor from within python'''
    
def run_sex(name):
    
    os.system("sex -c imaging/default_"+name+".sex "+"imaging/pet_radius_"+name+".fits")
    

'''Extracts parameters from source extractor that are useful as a Galfit input/ first guess'''
def sex_params_galfit(name):
    
    print(os.getcwd())
    file_path = 'imaging/test_'+name+'.cat'
    f = open(file_path, 'r+')
    data=f.readlines()
    A=[]
    B=[]
    B_mult=[]
    pet_mult=[]
    flux_pet=[]
    x_pos=[]
    y_pos=[]
    mag_auto=[]
    eff_radius=[]
    PA=[]
    back=[]
    for line in data:
        words = line.split()
        if words[0] !='#':
            x_pos.append(float(words[1]))
            y_pos.append(float(words[2]))
            A.append(float(words[3]))#major axis RMS
            B.append(float(words[4]))#minor axis RMS
            eff_radius.append(float(words[7]))#this is the effective radius
            pet_mult.append(float(words[8]))#this is the petrosian radius
            flux_pet.append(float(words[10]))#flux within the petrosian aperture in counts
            PA.append(float(words[12]))#position angle of the source
            back.append(float(words[13]))#background at centroid position in counts
    
    max_i=flux_pet.index(max(flux_pet))
    x_max=x_pos[max_i]
    y_max=y_pos[max_i]
    sex_pet_r=(pet_mult[max_i]*A[max_i])
    minor=pet_mult[max_i]*B[max_i]
    flux=flux_pet[max_i]
    
    eff_radius_1=eff_radius[max_i]
    B_A_1=B[max_i]/A[max_i]#elipticity
    PA_1=PA[max_i]
    back_1=back[max_i]
    
    
    if len(x_pos)==1:#This means there was only one aperture detected therefore one stellar bulge
        n_bulges=1
        return sex_pet_r, minor, flux, x_max, y_max, n_bulges, eff_radius_1, B_A_1, PA_1, back_1
    else:
        import heapq
        two_largest=heapq.nlargest(2, flux_pet)
        if two_largest[1] > 0.1*two_largest[0]:
            #then it's 1/10th of the brightness and is legit
            n_bulges=2
            max_i_sec=flux_pet.index(two_largest[1])
            x_max_2=x_pos[max_i_sec]
            y_max_2=y_pos[max_i_sec]
            flux_max_2=flux_pet[max_i_sec]
            eff_radius_2=eff_radius[max_i_sec]
            B_A_2=B[max_i_sec]/A[max_i_sec]
            PA_2=PA[max_i_sec] #was -90-PA[max_i]+180
            return sex_pet_r, minor, flux, x_max, y_max, n_bulges, eff_radius_1, B_A_1, PA_1, back_1, x_max_2, y_max_2, flux_max_2, eff_radius_2, B_A_2, PA_2
        else:
            n_bulges=1
            return sex_pet_r, minor, flux, x_max, y_max, n_bulges, eff_radius_1, B_A_1, PA_1, back_1
    
'''Optionally plots the source extractor fit and captures the background'''
def plot_sex(name,xpos,ypos):
    pic='imaging/aps_'+name+'.fits'
    im=pyfits.open(pic)
    #optional plotting of the Source Extractor output:
    plt.clf()
    im=pyfits.open(pic)
    plt.title('Sex Apertures')
    plt.imshow(im[0].data,norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.scatter(xpos,ypos, color='black')
    plt.savefig('sdss/sex_aps_'+name+'.pdf')
    return im[0].data#this is the bg image
   

'''Writes the galfit feedme file'''
def write_galfit_feedme(name,xcen,ycen,xcen2,ycen2, mag, mag_zpt, num_bulges, length_gal, r_1, r_2, mag_2, B_A_1, PA_1, B_A_2, PA_2, background):
    if num_bulges==2: 
    
        '''I need to make a code to write out the GALFIT.feedme file'''
        file = open('imaging/galfit.feedme_'+str(name), "w")
        file.write('==============================================================================='+'\n')

        file.write('# IMAGE and GALFIT CONTROL PARAMETERS'+'\n')


        file.write('A) imaging/out_convolved_'+str(name)+'.fits            # Input data image (FITS file)'+'\n')
        file.write('B) imaging/out_'+str(name)+'.fits       # Output data image block'+'\n')
        file.write('C) imaging/out_sigma_convolved_'+str(name)+'.fits                # Sigma image name (made from data i$'+'\n')
        file.write('D) imaging/psf_'+str(name)+'.fits   #        # Input PSF image and (optional) diffusion kernel'+'\n')
        file.write('E) none                   # PSF fine sampling factor relative to data'+'\n')
        file.write('F) none                # Bad pixel mask (FITS image or ASCII coord list)'+'\n')
        file.write('G) none                # File with parameter constraints (ASCII file)'+'\n')
        file.write('H) 1    '+str(length_gal)+'   1    '+str(length_gal)+'   # Image region to fit (xmin xmax ymin ymax)'+'\n')
        file.write('I) '+str(length_gal)+' '+ str(length_gal)+'          # Size of the convolution box (x y)'+'\n')
        file.write('J) '+str(mag_zpt)+' # Magnitude photometric zeropoint'+'\n')
        file.write('K) 0.5  0.5        # Plate scale (dx dy)    [arcsec per pixel]'+'\n')
        file.write('O) regular             # Display type (regular, curses, both)'+'\n')
        file.write('P) 0                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps'+'\n')

        #first bulge
        file.write('# Object number: 1 '+'\n')
        file.write(' 0) sersic                 #  object type'+'\n')
        file.write(' 1) '+str(xcen)+' '+str(ycen)+'  1 1  #  position x, y'+'\n')#these positions need to be automated
        file.write(' 3) '+str(mag)+'    1          #  Integrated magnitude'+'\n')#'+str(mag)+'
        file.write(' 4) '+str(r_1)+'      1          #  R_e (half-light radius)   [pix]'+'\n')
        file.write(' 5) 4      1          #  Sersic index n (de Vaucouleurs n=4)'+'\n')
        file.write(' 6) 0.0000      0          #     -----'+'\n')
        file.write(' 7) 0.0000      0          #     -----'+'\n')
        file.write(' 8) 0.0000      0          #     -----'+'\n')
        file.write(' 9) '+str(B_A_1)+'      1          #  axis ratio (b/a)'+'\n')
        file.write('10) '+str(PA_1+90)+'    1          #  position angle (PA) [deg: Up=0, Left=90]'+'\n')
        file.write(' Z) 0                      #  output option (0 = resid., 1 = Dont subtract)'+'\n')

        
        file.write('# Object number: 2 '+'\n')
        file.write(' 0) sersic                 #  object type'+'\n')
        file.write(' 1) '+str(xcen2)+' '+str(ycen2)+'  1 1  #  position x, y'+'\n')
        file.write(' 3) '+str(mag_2)+'    1          #  Integrated magnitude'+'\n')#'+str(mag)+'
        file.write(' 4) '+str(r_2)+'      1          #  R_e (half-light radius)   [pix]'+'\n')
        file.write(' 5) 4      1          #  Sersic index n (de Vaucouleurs n=4)'+'\n')
        file.write(' 6) 0.0000      0          #     -----'+'\n')
        file.write(' 7) 0.0000      0          #     -----'+'\n')
        file.write(' 8) 0.0000      0          #     -----'+'\n')
        file.write(' 9) '+str(B_A_2)+'      1          #  axis ratio (b/a)'+'\n')
        file.write('10) '+str(PA_2+90)+'    1          #  position angle (PA) [deg: Up=0, Left=90]'+'\n')
        file.write(' Z) 0                      #  output option (0 = resid., 1 = Dont subtract)'+'\n')

        '''file.write('# Object number: 3 '+'\n')
        file.write(' 0) sersic                 #  object type'+'\n')
        file.write(' 1) '+str(xcen)+' '+str(ycen)+'  1 1  #  position x, y'+'\n')#these positions need to be automated                                                                                 
        file.write(' 3) '+str(mag)+'    1          #  Integrated magnitude'+'\n')#'+str(mag)+'                                                                                                         
        file.write(' 4) '+str(r_1)+'      1          #  R_e (half-light radius)   [pix]'+'\n')
        file.write(' 5) 1      1          #  Sersic index n (de Vaucouleurs n=4)'+'\n')
        file.write(' 6) 0.0000      0          #     -----'+'\n')
        file.write(' 7) 0.0000      0          #     -----'+'\n')
        file.write(' 8) 0.0000      0          #     -----'+'\n')
        file.write(' 9) '+str(B_A_1)+'      1          #  axis ratio (b/a)'+'\n')
        file.write('10) '+str(PA_1+90)+'    1          #  position angle (PA) [deg: Up=0, Left=90]'+'\n')
        file.write(' Z) 0                      #  output option (0 = resid., 1 = Dont subtract)'+'\n')'''
    
    
        file.write('# Object number: 4'+'\n')
        file.write(' 0) sky                    #  object type'+'\n')
        file.write(' 1) '+str(background)+'      1          #  sky background at center of fitting region [ADU$'+'\n')
        #was 100.3920      1    
        file.write(' 2) 0.0000      0          #  dsky/dx (sky gradient in x)'+'\n')
        file.write(' 3) 0.0000      0          #  dsky/dy (sky gradient in y)'+'\n')
        file.write(' Z) 0                      #  output option (0 = resid., 1 = Dont subtract)'+'\n')

        file.write('================================================================================'+'\n')
    if num_bulges==1:
        '''I need to make a code to write out the GALFIT.feedme file'''
        file = open('imaging/galfit.feedme_'+str(name), "w")
        file.write('==============================================================================='+'\n')

        file.write('# IMAGE and GALFIT CONTROL PARAMETERS'+'\n')


        file.write('A) imaging/out_convolved_'+str(name)+'.fits            # Input data image (FITS file)'+'\n')
        file.write('B) imaging/out_'+str(name)+'.fits       # Output data image block'+'\n')
        file.write('C) imaging/out_sigma_convolved_'+str(name)+'.fits                # Sigma image name (made from data i$'+'\n')
        file.write('D) imaging/psf_'+str(name)+'.fits   #        # Input PSF image and (optional) diffusion kernel'+'\n')
        file.write('E) none                   # PSF fine sampling factor relative to data'+'\n')
        file.write('F) none                # Bad pixel mask (FITS image or ASCII coord list)'+'\n')
        file.write('G) none                # File with parameter constraints (ASCII file)'+'\n')
        file.write('H) 1    '+str(length_gal)+'   1    '+str(length_gal)+'   # Image region to fit (xmin xmax ymin ymax)'+'\n')
        file.write('I) '+str(length_gal)+' '+ str(length_gal)+'          # Size of the convolution box (x y)'+'\n')
        file.write('J) '+str(mag_zpt)+' # Magnitude photometric zeropoint'+'\n')
        
        file.write('K) 0.5  0.5        # Plate scale (dx dy)    [arcsec per pixel]'+'\n')
        file.write('O) regular             # Display type (regular, curses, both)'+'\n')
        file.write('P) 0                   # Choose: 0=optimize, 1=model, 2=imgblock, 3=subcomps'+'\n')

        #first bulge
        file.write('# Object number: 1 '+'\n')
        file.write(' 0) sersic                 #  object type'+'\n')
        file.write(' 1) '+str(xcen)+' '+str(ycen)+'  1 1  #  position x, y'+'\n')#these positions need to be automated
        file.write(' 3) '+str(mag)+'     1          #  Integrated magnitude'+'\n')#'+str(mag)+'
        file.write(' 4) '+str(r_1)+'      1          #  R_e (half-light radius)   [pix]'+'\n')
        file.write(' 5) 4      1          #  Sersic index n (de Vaucouleurs n=4)'+'\n')
        file.write(' 6) 0.0000      0          #     -----'+'\n')
        file.write(' 7) 0.0000      0          #     -----'+'\n')
        file.write(' 8) 0.0000      0          #     -----'+'\n')
        file.write(' 9) '+str(B_A_1)+'     1          #  axis ratio (b/a)'+'\n')
        file.write('10) '+str(PA_1+90)+'    1          #  position angle (PA) [deg: Up=0, Left=90]'+'\n')
        file.write(' Z) 0                      #  output option (0 = resid., 1 = Dont subtract)'+'\n')

        '''file.write('# Object number: 2 '+'\n')
        file.write(' 0) sersic                 #  object type'+'\n')
        file.write(' 1) '+str(xcen)+' '+str(ycen)+'  1 1  #  position x, y'+'\n')#these positions need to be automated                                                                                 
        file.write(' 3) '+str(mag)+'     1          #  Integrated magnitude'+'\n')#'+str(mag)+'                                                                                                        
        file.write(' 4) '+str(r_1)+'      1          #  R_e (half-light radius)   [pix]'+'\n')
        file.write(' 5) 1      1          #  Sersic index n (de Vaucouleurs n=4)'+'\n')
        file.write(' 6) 0.0000      0          #     -----'+'\n')
        file.write(' 7) 0.0000      0          #     -----'+'\n')
        file.write(' 8) 0.0000      0          #     -----'+'\n')
        file.write(' 9) '+str(B_A_1)+'     1          #  axis ratio (b/a)'+'\n')
        file.write('10) '+str(PA_1+90)+'    1          #  position angle (PA) [deg: Up=0, Left=90]'+'\n')
        file.write(' Z) 0                      #  output option (0 = resid., 1 = Dont subtract)'+'\n')'''
        
        file.write('# Object number: 3'+'\n')
        file.write(' 0) sky                    #  object type'+'\n')
        file.write(' 1) '+str(background)+'      1          #  sky background at center of fitting region [ADU$'+'\n')
        #was 100.3920      1                                                                                                                                                                           
        file.write(' 2) 0.0000      0          #  dsky/dx (sky gradient in x)'+'\n')
        file.write(' 3) 0.0000      0          #  dsky/dy (sky gradient in y)'+'\n')
        file.write(' Z) 0                      #  output option (0 = resid., 1 = Dont subtract)'+'\n')

        file.write('================================================================================'+'\n')



    file.close()
    
    

          



'''
Calls galfit with the feedme file you created
'''
import subprocess
def run_galfit(name):
    
    call ="~/Applications/galfit imaging/galfit.feedme_"+str(name)+">/dev/null 2>&1"
    #print('call', call)
    #print(shlex.split(call))
    #subprocess.check_output(shlex.split(call), shell=True)
    subprocess.call(call,shell=True)
    
    
    
'''Now open the galfit result and get the effective radius
I've kept a lot of unnecessary stuff in here in case you want to
use additional pieces of the output; this code is generally
very useful for running galfit inline and extracting the output info'''
def galfit_params(name,num_bulges,kpcpix):
    
    '''First, figure out how many bulges'''
 
    plt.clf()
    try:
        output='imaging/out_'+str(name)+'.fits'
        out=pyfits.open(output)
    except FileNotFoundError:
        return None
    
    fig=plt.figure()
    ax1=fig.add_subplot(211)
    im1=ax1.imshow(out[1].data, norm=matplotlib.colors.LogNorm())
    plt.colorbar(im1,label='Magnitudes')
    ax2=fig.add_subplot(212)
    im2=ax2.imshow(out[2].data,norm=matplotlib.colors.LogNorm())
    plt.colorbar(im2,label='Magnitudes')
    plt.savefig('imaging/side_by_side_galfit_input_'+str(name)+'.pdf')
    
    
    if num_bulges==1:
        
        mag_1=float(out[2].header['1_MAG'][:7])

        sep=0
        flux_ratio=0
        x_1=float(out[2].header['1_XC'][:7])
        x_2=x_1
        y_1=float(out[2].header['1_YC'][:7])
        y_2=y_1
        PA1=float(out[2].header['1_PA'][:7])
        PA2=PA1
#        AR1=float(out[2].header['1_AR'][:7])
        try:
            AR1=float(out[2].header['1_AR'][:7])
            sersic = float(out[2].header['1_N'][:7])
        except ValueError or KeyError:
            try:
                sersic=float(out[2].header['1_N'][1:7])
            except ValueError or KeyError:
                    return 0,0,999        

        
    else:
        try:

            mag_1=float(out[2].header['1_MAG'][:7])
            mag_2=float(out[2].header['2_MAG'][:7])

            #m = -2.5 log_10 (F/ t) + (mag zpt = 26.563)
            #t*10^((m-zpt)/(-2.5)) = F
            #F/F = 10^(m-m/-2.5)
            
            #The flux ratio:
            flux_ratio=10**((mag_1-mag_2)/-2.5)

            
            
            if mag_1 < mag_2:
                #this means that #1 is brighter
                x_1=float(out[2].header['1_XC'][:7])
                x_2=float(out[2].header['2_XC'][:7])
                y_1=float(out[2].header['1_YC'][:7])
                y_2=float(out[2].header['2_YC'][:7])
                PA1=float(out[2].header['1_PA'][:7])
                PA2=float(out[2].header['2_PA'][:7])
                AR1=float(out[2].header['1_AR'][:7])
                sersic=float(out[2].header['1_N'][:7])
            else:
                x_1=float(out[2].header['2_XC'][:7])
                x_2=float(out[2].header['1_XC'][:7])
                y_1=float(out[2].header['2_YC'][:7])
                y_2=float(out[2].header['1_YC'][:7])
                PA1=float(out[2].header['2_PA'][:7])
                PA2=float(out[2].header['1_PA'][:7])
                AR1=float(out[2].header['2_AR'][:7])
                sersic=float(out[2].header['2_N'][:7])
                
            #Also, the separation in physical space between the bulges:
            sep=kpcpix*np.sqrt(abs(x_1-x_2)**2+abs(y_1-y_2)**2)
        except ValueError or KeyError:
            
            try:
                mag_1=float(out[2].header['1_MAG'][:7])
                sep=0
                flux_ratio=0
                flux_ratio=0
                x_1=float(out[2].header['1_XC'][:7])
                x_2=x_1
                y_1=float(out[2].header['1_YC'][:7])
                y_2=y_1
                PA1=float(out[2].header['1_PA'][:7])
                PA2=PA1
                AR1=float(out[2].header['1_AR'][:7])
                sersic=float(out[2].header['1_N'][:7])
            except ValueError or KeyError:
                try:
                    mag_1=float(out[2].header['2_MAG'][:7])
                    sep=0
                    flux_ratio=0
                    flux_ratio=0
                    x_1=float(out[2].header['2_XC'][:7])
                    x_2=x_1
                    y_1=float(out[2].header['2_YC'][:7])
                    y_2=y_1
                    PA1=float(out[2].header['2_PA'][:7])
                    PA2=PA1
                    AR1=float(out[2].header['2_AR'][:7])
                    sersic=float(out[2].header['2_N'][:7])
                except ValueError or KeyError:
                    return 0,0,999
                           
        

    galfit_input=out[1].data
    galfit_model=out[2].data
    galfit_resid=out[3].data
    
    
    chi2=(out[2].header['CHI2NU'])
    #print('Sersic', sersic)
    
    
    return sep, flux_ratio, sersic, AR1








'''Uses the Petrosian radius measured with Source Extractor and the semimajor axis and the petrosian flux
to calculate the surface brightness at the petrosian radius which we then will use to construct the segmentation
maps in a later step'''
def input_petrosian_radius(input_r, input_b, flux_w_i):
    
    Area=(input_b*input_r*np.pi) #area in pixels of an ellipse
    limiting_sb=0.2*(flux_w_i/Area)#was 0.2*(flux_w_i/Area)
    
    
    return limiting_sb#returns the surface brightness at the petrosian radius



def img_assy_shape(img,mangaid):

    gal_zoom=img
        
    '''
    Now we need to do 8-connectivity:
    Basically from the center out, is something really 8-connected?
    '''
    
    half_bit=int(np.shape(gal_zoom)[0]/2)
    
    
    
    labeled = measure.label(gal_zoom, background=False, connectivity=2)
    label = labeled[half_bit, half_bit] # known pixel location

    rp = measure.regionprops(labeled)
    props = rp[label - 1] # background is labeled 0, not in rp

    props.bbox # (min_row, min_col, max_row, max_col)
    gal_zoom=props.image
    
    
    
    
    
    gal_zoom=gal_zoom.astype(np.float32)
    
    A=np.sum(abs(gal_zoom-np.rot90(np.rot90(gal_zoom))))/np.sum(abs(gal_zoom))
    plt.clf()
    plt.imshow(abs(gal_zoom-np.rot90(np.rot90(gal_zoom))))
    plt.title(A)
    plt.savefig('imaging/SA_'+str(mangaid)+'.pdf')
        
    return A




"""
This file defines the `make_figure` function, which can be useful for
debugging and/or examining the morphology of a source in detail.
"""
# Author: Vicente Rodriguez-Gomez <v.rodriguez@irya.unam.mx>
# Licensed under a 3-Clause BSD License.

import numpy as np
import warnings
import time
import sys
if 'matplotlib' not in sys.modules:
    import matplotlib
    if sys.version_info[0] == 2:  # Python 2
        matplotlib.use('agg')
    elif sys.version_info[0] == 3:  # Python 3
        matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.cm
import scipy.signal
import scipy.ndimage as ndi
import skimage.transform
from astropy.io import fits
from astropy.visualization import LogStretch

__all__ = ['make_figure']

def normalize(image, m=None, M=None):
    if m is None:
        m = np.min(image)
    if M is None:
        M = np.max(image)

    retval = (image-m) / (M-m)
    retval[image <= m] = 0.0
    retval[image >= M] = 1.0

    return retval

def get_ax(fig, row, col, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig):
    x_ax = (col+1)*eps + col*wpanel
    y_ax = eps + (nrows-1-row)*(hpanel+htop)
    return fig.add_axes([x_ax/wfig, y_ax/hfig, wpanel/wfig, hpanel/hfig])

def make_figure(morph):
    """
    Creates a figure analogous to Fig. 4 from Rodriguez-Gomez et al. (2018)
    for a given ``SourceMorphology`` object.
    
    Parameters
    ----------
    morph : ``statmorph.SourceMorphology``
        An object containing the morphological measurements of a single
        source.
    Returns
    -------
    fig : ``matplotlib.figure.Figure``
        The figure.
    """
    # I'm tired of dealing with plt.add_subplot, plt.subplots, plg.GridSpec,
    # plt.subplot2grid, etc. and never getting the vertical and horizontal
    # inter-panel spacings to have the same size, so instead let's do
    # everything manually:
    nrows = 2
    ncols = 4
    wpanel = 4.0  # panel width
    hpanel = 4.0  # panel height
    htop = 0.05*nrows*hpanel  # top margin and vertical space between panels
    eps = 0.005*nrows*hpanel  # all other margins
    wfig = ncols*wpanel + (ncols+1)*eps  # total figure width
    hfig = nrows*(hpanel+htop) + eps  # total figure height
    fig = plt.figure(figsize=(wfig, hfig))

    # For drawing circles/ellipses
    theta_vec = np.linspace(0.0, 2.0*np.pi, 200)

    # Add black to pastel colormap
    cmap_orig = matplotlib.cm.Pastel1
    colors = ((0.0, 0.0, 0.0), *cmap_orig.colors)
    cmap = matplotlib.colors.ListedColormap(colors)

    log_stretch = LogStretch(a=10000.0)

    # Get some general info about the image
    image = np.float64(morph._cutout_stamp_maskzeroed)  # skimage wants double
    ny, nx = image.shape
    m = np.min(image)
    M = np.max(image)
    m_stretch, M_stretch = log_stretch([m, M])
    xc, yc = morph._xc_stamp, morph._yc_stamp  # centroid
    xca, yca = morph._asymmetry_center  # asym. centroid

    ##################
    # Original image #
    ##################
    ax = get_ax(fig, 0, 0, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    ax.imshow(log_stretch(normalize(image, m=m, M=M)), cmap='gray', origin='lower',
              vmin=m_stretch, vmax=M_stretch)

    ax.plot(xc, yc, 'go', markersize=5, label='Centroid')
    R = float(nx**2 + ny**2)
    theta = morph.orientation_centroid
    x0, x1 = xc - R*np.cos(theta), xc + R*np.cos(theta)
    y0, y1 = yc - R*np.sin(theta), yc + R*np.sin(theta)
    ax.plot([x0, x1], [y0, y1], 'g--', lw=1.5, label='Major Axis (Centroid)')
    ax.plot(xca, yca, 'bo', markersize=5, label='Asym. Center')
    R = float(nx**2 + ny**2)
    theta = morph.orientation_asymmetry
    x0, x1 = xca - R*np.cos(theta), xca + R*np.cos(theta)
    y0, y1 = yca - R*np.sin(theta), yca + R*np.sin(theta)
    ax.plot([x0, x1], [y0, y1], 'b--', lw=1.5, label='Major Axis (Asym.)')
    # Half-radius ellipse
    a = morph.rhalf_ellip
    b = a / morph.elongation_asymmetry
    theta = morph.orientation_asymmetry
    xprime, yprime = a*np.cos(theta_vec), b*np.sin(theta_vec)
    x = xca + (xprime*np.cos(theta) - yprime*np.sin(theta))
    y = yca + (xprime*np.sin(theta) + yprime*np.cos(theta))
    ax.plot(x, y, 'b', label='Half-Light Ellipse')
    # Some text
    text = 'flag = %d\nEllip. (Centroid) = %.4f\nEllip. (Asym.) = %.4f' % (
        morph.flag, morph.ellipticity_centroid, morph.ellipticity_asymmetry)
    ax.text(0.034, 0.966, text,
        horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Finish plot
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_title('Original Image (Log Stretch)', fontsize=14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ##############
    # Sersic fit #
    ##############
    ax = get_ax(fig, 0, 1, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    y, x = np.mgrid[0:ny, 0:nx]
    sersic_model = morph._sersic_model(x, y)
    # Add background noise (for realism)
    if morph.sky_sigma > 0:
        sersic_model += np.random.normal(scale=morph.sky_sigma, size=(ny, nx))
    ax.imshow(log_stretch(normalize(sersic_model, m=m, M=M)), cmap='gray', origin='lower',
              vmin=m_stretch, vmax=M_stretch)
    # Sersic center (within postage stamp)
    xcs, ycs = morph._sersic_model.x_0.value, morph._sersic_model.y_0.value
    ax.plot(xcs, ycs, 'ro', markersize=5, label='Sérsic Center')
    R = float(nx**2 + ny**2)
    theta = morph.sersic_theta
    x0, x1 = xcs - R*np.cos(theta), xcs + R*np.cos(theta)
    y0, y1 = ycs - R*np.sin(theta), ycs + R*np.sin(theta)
    ax.plot([x0, x1], [y0, y1], 'r--', lw=1.5, label='Major Axis (Sérsic)')
    # Half-radius ellipse
    a = morph.sersic_rhalf
    b = a * (1.0 - morph.sersic_ellip)
    xprime, yprime = a*np.cos(theta_vec), b*np.sin(theta_vec)
    x = xc + (xprime*np.cos(theta) - yprime*np.sin(theta))
    y = yc + (xprime*np.sin(theta) + yprime*np.cos(theta))
    ax.plot(x, y, 'r', label='Half-Light Ellipse (Sérsic)')
    # Some text
    text = ('flag_sersic = %d' % (morph.flag_sersic) + '\n' +
            'Ellip. (Sérsic) = %.4f' % (morph.sersic_ellip) + '\n' +
            r'$n = %.4f$' % (morph.sersic_n))
    ax.text(0.034, 0.966, text,
        horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Finish plot
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_title('Sérsic Model + Noise', fontsize=14)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ###################
    # Sersic residual #
    ###################
    ax = get_ax(fig, 0, 2, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    y, x = np.mgrid[0:ny, 0:nx]
    sersic_res = morph._cutout_stamp_maskzeroed - morph._sersic_model(x, y)
    sersic_res[morph._mask_stamp] = 0.0
    ax.imshow(normalize(sersic_res), cmap='gray', origin='lower')
    ax.set_title('Sérsic Residual, ' + r'$I - I_{\rm model}$', fontsize=14)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ######################
    # Asymmetry residual #
    ######################
    ax = get_ax(fig, 0, 3, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    # Rotate image around asym. center
    image_180 = skimage.transform.rotate(image, 180.0, center=(xca, yca))
    image_res = image - image_180
    # Apply symmetric mask
    mask = morph._mask_stamp.copy()
    mask_180 = skimage.transform.rotate(mask, 180.0, center=(xca, yca))
    mask_180 = mask_180 >= 0.5  # convert back to bool
    mask_symmetric = mask | mask_180
    image_res = np.where(~mask_symmetric, image_res, 0.0)
    ax.imshow(normalize(image_res), cmap='gray', origin='lower')
    ax.set_title('Asymmetry Residual, ' + r'$I - I_{180}$', fontsize=14)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ###################
    # Original segmap #
    ###################
    ax = get_ax(fig, 1, 0, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    ax.imshow(log_stretch(normalize(image, m=m, M=M)), cmap='gray', origin='lower',
              vmin=m_stretch, vmax=M_stretch)
    # Show original segmap
    contour_levels = [0.5]
    contour_colors = [(0,0,0)]
    segmap_stamp = morph._segmap.data[morph._slice_stamp]
    Z = np.float64(segmap_stamp == morph.label)
    C = ax.contour(Z, contour_levels, colors=contour_colors, linewidths=1.5)
    # Show skybox
    xmin = morph._slice_skybox[1].start
    ymin = morph._slice_skybox[0].start
    xmax = morph._slice_skybox[1].stop - 1
    ymax = morph._slice_skybox[0].stop - 1
    ax.plot(np.array([xmin, xmax, xmax, xmin, xmin]) + 0.5,
            np.array([ymin, ymin, ymax, ymax, ymin]) + 0.5,
            'b', lw=1.5, label='Skybox')
    # Some text
    text = ('Sky Mean = %.4f' % (morph.sky_mean) + '\n' +
            'Sky Median = %.4f' % (morph.sky_median) + '\n' +
            'Sky Sigma = %.4f' % (morph.sky_sigma)+'\n'+
            'Sky min = %.4f' % (m) + '\n' +
            'Sky max = %.4f' % (M))
    ax.text(0.034, 0.966, text,
        horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Finish plot
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_title('Original Segmap', fontsize=14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ###############
    # Gini segmap #
    ###############
    ax = get_ax(fig, 1, 1, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    ax.imshow(log_stretch(normalize(image, m=m, M=M)),
              cmap='gray', origin='lower', vmin=m_stretch, vmax=M_stretch)
    # Show Gini segmap
    contour_levels = [0.5]
    contour_colors = [(1,0,1)]
    Z = np.float64(morph._segmap_gini)
    C = ax.contour(Z, contour_levels, colors=contour_colors, linewidths=1.5)
    # Some text
    text = r'$\left\langle {\rm S/N} \right\rangle = %.4f$' % (morph.sn_per_pixel)
    ax.text(0.034, 0.966, text, fontsize=12,
        horizontalalignment='left', verticalalignment='top',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    text = (r'$G = %.4f$' % (morph.gini) + '\n' +
            r'$M_{20} = %.4f$' % (morph.m20) + '\n' +
            r'$F(G, M_{20}) = %.4f$' % (morph.gini_m20_bulge) + '\n' +
            r'$S(G, M_{20}) = %.4f$' % (morph.gini_m20_merger))
    ax.text(0.034, 0.034, text, fontsize=12,
        horizontalalignment='left', verticalalignment='bottom',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    text = (r'$C = %.4f$' % (morph.concentration) + '\n' +
            r'$A = %.4f$' % (morph.asymmetry) + '\n' +
            r'$S = %.4f$' % (morph.smoothness))
    ax.text(0.966, 0.034, text, fontsize=12,
        horizontalalignment='right', verticalalignment='bottom',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    # Finish plot
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_title('Gini Segmap', fontsize=14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ####################
    # Watershed segmap #
    ####################
    ax = get_ax(fig, 1, 2, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    labeled_array, peak_labels, xpeak, ypeak = morph._watershed_mid
    labeled_array_plot = (labeled_array % (cmap.N-1)) + 1
    labeled_array_plot[labeled_array == 0] = 0.0  # background is black
    ax.imshow(labeled_array_plot, cmap=cmap, origin='lower',
              norm=matplotlib.colors.NoNorm())
    sorted_flux_sums, sorted_xpeak, sorted_ypeak = morph._intensity_sums
    if len(sorted_flux_sums) > 0:
        ax.plot(sorted_xpeak[0] + 0.5, sorted_ypeak[0] + 0.5, 'bo', markersize=2, label='First Peak')
    if len(sorted_flux_sums) > 1:
        ax.plot(sorted_xpeak[1] + 0.5, sorted_ypeak[1] + 0.5, 'ro', markersize=2, label='Second Peak')
    # Some text
    text = (r'$M = %.4f$' % (morph.multimode) + '\n' +
            r'$I = %.4f$' % (morph.intensity) + '\n' +
            r'$D = %.4f$' % (morph.deviation))
    ax.text(0.034, 0.034, text, fontsize=12,
        horizontalalignment='left', verticalalignment='bottom',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_title('Watershed Segmap (' + r'$I$' + ' statistic)', fontsize=14)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ##########################
    # Shape asymmetry segmap #
    ##########################
    ax = get_ax(fig, 1, 3, nrows, ncols, wpanel, hpanel, htop, eps, wfig, hfig)
    ax.imshow(morph._segmap_shape_asym, cmap='gray', origin='lower')
    ax.plot(xca, yca, 'bo', markersize=5, label='Asym. Center')
    r = morph.rpetro_circ
    ax.plot(xca + r*np.cos(theta_vec), yca + r*np.sin(theta_vec), 'b', label=r'$r_{\rm petro, circ}$')
    r = morph.rpetro_ellip
    ax.plot(xca + r*np.cos(theta_vec), yca + r*np.sin(theta_vec), 'r', label=r'$r_{\rm petro, ellip}$')
    r = morph.rmax_circ
    ax.plot(np.floor(xca) + r*np.cos(theta_vec), np.floor(yca) + r*np.sin(theta_vec), 'c', lw=1.5, label=r'$r_{\rm max}$')
    # ~ r = morph._petro_extent_flux * morph.rpetro_ellip
    # ~ ax.plot(xca + r*np.cos(theta_vec), yca + r*np.sin(theta_vec), 'r--', label='%g*rpet_ellip' % (morph._petro_extent_flux))
    text = (r'$A_S = %.4f$' % (morph.shape_asymmetry))
    ax.text(0.034, 0.034, text, fontsize=12,
        horizontalalignment='left', verticalalignment='bottom',
        transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=1.0, boxstyle='round'))
    ax.legend(loc=4, fontsize=12, facecolor='w', framealpha=1.0, edgecolor='k')
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_title('Shape Asymmetry Segmap', fontsize=14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # defaults: left = 0.125, right = 0.9, bottom = 0.1, top = 0.9, wspace = 0.2, hspace = 0.2
    fig.subplots_adjust(left=eps/wfig, right=1-eps/wfig, bottom=eps/hfig, top=1.0-htop/hfig, wspace=eps/wfig, hspace=htop/hfig)

    #fig.savefig('test_segmap.png', dpi=150)
    
    return fig
 
'''
Now we begin.
'''

# The first step is to import the necessary information from 'table_manga_gals.txt'

# Import it as a dataframe


feature_dict = {i:label for i,label in zip(
range(7),
  ('PLATEIFU',
  'run',
  'field',
  'camcol',
  'RA',
'DEC','redshift'))}

df = pd.io.parsers.read_csv(filepath_or_buffer='table_manga_gals.txt',#'_view_all.txt',#filepath_or_buffer='LDA_img_ratio_'+str(run)+'_early_late_all_things.txt',#'_view_all.txt',
       header=[0],
       sep='\t'
       )
df.columns = [l for i,l in sorted(feature_dict.items())]

df.dropna(how="all", inplace=True) # to drop the empty line at file-end

print(df)



#SDSS superclean merger Galaxy Zoo
#sdss_list=[587730773351923978, 587727177912680538, 587727223561257225, 587731187277955213, 587730774962864138, 587727178986750088, 588015510343712908, 587730775499997384, 587727180060688562, 587727225690783827, 587727226227720294, 587731186204868616, 588015510344040607, 588015508733624406, 587727225691242683, 587730772816167051, 588015510344302646, 588015509807497271, 587727178450927744, 588290881638760664, 587727178988191864]
#ra_list=['00:00:20.24', '00:00:37.17', '00:02:27.29', '00:02:49.07', '00:03:08.23', '00:03:57.92', '00:05:27.39', '00:05:54.09', '00:05:58.00', '00:06:07.51', '00:06:24.76', '00:08:27.81', '00:08:32.88', '00:10:11.15', '00:10:33.29', '00:10:55.50', '00:11:08.83', '00:11:43.68', '00:13:30.75', '00:14:31.15', '00:16:54.99']
#dec_list=['+14:11:09.9', '-11:02:07.6', '+16:04:42.5', '+00:45:04.8', '+15:33:48.8', '-10:20:44.3', '+00:50:48.1', '+15:53:42.8', '-09:21:17.8', '-10:30:32.7', '-10:01:31.2', '-00:00:17.7', '+01:02:20.1', '-00:14:30.8', '-10:36:10.9', '+13:52:49.7', '+00:50:43.7', '+00:31:22.5', '-10:43:17.6', '+15:49:02.2', '-10:23:44.0']


merger=np.zeros(len(df))
kept_ids=[]#if you need a list of which ones ran without failing

print(len(df))



'''
GalaxySmelter_Real.py:
This file will be your input to the MergerMonger code
It can be adjusted for use with different galaxy surveys.
Some inputs that are necessary to change are the camera_data
and camera_data_sigma and psf files, which are in counts.

Also, things to change for different surveys:
1. pixelscale (kpc/")
2. redshift of each galaxy
'''

file=open('LDA_img_statmorph_MaNGA_test.txt','w')



counter=0

mag_list=[]
S_N_list=[]

for i in range(len(df)):
    sdss=df['PLATEIFU'][i]
    #i=i+1
    '''Search through drpall for redshift, the mangaid, and the designid, you need these to directly
    access the preimaging files, which are not stored in the same format as our input lists'''
    
    
    #decode=SDSS_objid_to_values(sdss)
    #return skyVersion, rerun, run, camcol, field, object_num
    #https://data.sdss.org/sas/dr14/eboss/photoObj/frames/301/2566/2/

    print('run', df['run'][i], 'camcol', df['camcol'][i], 'field', df['field'][i], 'redshift', df['redshift'][i])
    
        
    try:
        if df['run'][i] < 1000:
            if df['camcol'][i] > 100:
                im=fits.open('sdss/frame-r-000'+str(df['run'][i])+'-'+str(df['field'][i])+'-0'+str(df['camcol'][i])+'.fits')
            if df['camcol'][i] < 100:
                im=fits.open('sdss/frame-r-000'+str(df['run'][i])+'-'+str(df['field'][i])+'-00'+str(df['camcol'][i])+'.fits')
        else:
            if df['camcol'][i] > 100:
                im=fits.open('sdss/frame-r-00'+str(df['run'][i])+'-'+str(df['field'][i])+'-0'+str(df['camcol'][i])+'.fits')
            if df['camcol'][i] < 100:
                im=fits.open('sdss/frame-r-00'+str(df['run'][i])+'-'+str(df['field'][i])+'-00'+str(df['camcol'][i])+'.fits')
    except FileNotFoundError:
        os.chdir(os.path.expanduser('/Users/beckynevin/CfA_Code/Kinematics_and_Imaging_Merger_Identification/Identify_MaNGA_SDSS_mergers_imaging/sdss'))
        '''Here is where it is necessary to know the SDSS data password and username'''
        try:
            if df['run'][i] < 1000:
                if df['camcol'][i] > 100:
                    os.system('wget https://data.sdss.org/sas/dr14/eboss/photoObj/frames/301/'+str(df['run'][i])+'/'+str(df['field'][i])+'/frame-r-000'+str(df['run'][i])+'-'+str(df['field'][i])+'-0'+str(df['camcol'][i])+'.fits.bz2')
                    os.system('gunzip frame-r-000'+str(df['run'][i])+'-'+str(df['field'][i])+'-0'+str(df['camcol'][i])+'.fits.bz2')
                if df['camcol'][i] < 100:
                    os.system('wget https://data.sdss.org/sas/dr14/eboss/photoObj/frames/301/'+str(df['run'][i])+'/'+str(df['field'][i])+'/frame-r-000'+str(df['run'][i])+'-'+str(df['field'][i])+'-00'+str(df['camcol'][i])+'.fits.bz2')
                    os.system('gunzip frame-r-000'+str(df['run'][i])+'-'+str(df['field'][i])+'-00'+str(df['camcol'][i])+'.fits.bz2')
            else:
                if df['camcol'][i] > 100:
                    os.system('wget https://data.sdss.org/sas/dr14/eboss/photoObj/frames/301/'+str(df['run'][i])+'/'+str(df['field'][i])+'/frame-r-00'+str(df['run'][i])+'-'+str(df['field'][i])+'-0'+str(df['camcol'][i])+'.fits.bz2')
                    os.system('gunzip frame-r-00'+str(df['run'][i])+'-'+str(df['field'][i])+'-0'+str(df['camcol'][i])+'.fits.bz2')
                if df['camcol'][i] < 100:
                    os.system('wget https://data.sdss.org/sas/dr14/eboss/photoObj/frames/301/'+str(df['run'][i])+'/'+str(df['field'][i])+'/frame-r-00'+str(df['run'][i])+'-'+str(df['field'][i])+'-00'+str(df['camcol'][i])+'.fits.bz2')
                    os.system('gunzip frame-r-00'+str(df['run'][i])+'-'+str(df['field'][i])+'-00'+str(df['camcol'][i])+'.fits.bz2')
        except FileNotFoundError:
            continue
        os.chdir(os.path.expanduser('/Users/beckynevin/CfA_Code/Kinematics_and_Imaging_Merger_Identification/Identify_MaNGA_SDSS_mergers_imaging'))


    try:
        if df['run'][i] < 1000:
            if df['camcol'][i] > 100:
                im=fits.open('sdss/frame-r-000'+str(df['run'][i])+'-'+str(df['field'][i])+'-0'+str(df['camcol'][i])+'.fits')
            if df['camcol'][i] < 100:
                im=fits.open('sdss/frame-r-000'+str(df['run'][i])+'-'+str(df['field'][i])+'-00'+str(df['camcol'][i])+'.fits')
        else:
            if df['camcol'][i] > 100:
                im=fits.open('sdss/frame-r-00'+str(df['run'][i])+'-'+str(df['field'][i])+'-0'+str(df['camcol'][i])+'.fits')
            if df['camcol'][i] < 100:
                im=fits.open('sdss/frame-r-00'+str(df['run'][i])+'-'+str(df['field'][i])+'-00'+str(df['camcol'][i])+'.fits')
    except FileNotFoundError:
        STOP
    
    #Now you need to cut out around the object
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    from astroquery.sdss import SDSS
    from astropy import coordinates as coords
    from astropy import units as u
    

    
    obj_coords = SkyCoord(str(df['RA'][i])+' '+str(df['DEC'][i]),unit=(u.hourangle, u.deg))

    #pos = coords.SkyCoord(str(ra_list[i])+' '+str(dec_list[i]), unit = u.deg, frame='fk5')#"1:12:43.2 +1:12:43", unit=u.deg
    #xid = SDSS.query_region(pos, spectro=True)
    #hdu = SDSS.get_images(matches=xid, band='r')[0][0]#[0][0]
    #STOP

    # Create 2D cutouts of the object in each band in a 6 by 6 arcsec box
    size = u.Quantity((80,80), u.arcsec)
    #print(np.shape(im[0].data))
    
    wcs_a = WCS(im[0].header)

    
    #print(wcs_a)

    from astropy.nddata import Cutout2D
    stamp_a = Cutout2D(im[0].data, obj_coords, size, wcs=wcs_a)#was image_a[0].data
    camera_data=(np.fliplr(np.rot90(stamp_a.data))/im[0].header['NMGY'])
    # Plot the the stamp in band Z with contours from band A
    #ax = plt.subplot(projection = wcs_a)
    #plt.imshow(camera_data*0.005, norm=matplotlib.colors.LogNorm(vmin=10**-1.5), cmap='afmhot')#, cmap='gray', origin='lower', alpha=1)
    #plt.colorbar()
    #plt.axis('off')
    #plt.savefig('sdss/'+str(sdss)+'.pdf')
    #continue

    plt.clf()
    plt.imshow(camera_data, norm=matplotlib.colors.LogNorm(vmin=10**0, vmax=10**4))
    #plt.title(np.min(AB_mag))
    plt.colorbar()
    plt.savefig('imaging/SDSS_counts_'+str(sdss)+'.png')


    '''Convert to magnitudes'''
    #units are nmgy, BUNIT, 1 nmgy = 3.631e-6 Jy
    #SDSS ugriz magnitudes are on the AB system where a magnitude 0 object has the same counts as a source with
    #F_nu = 3631 Jy
    Jy = (np.fliplr(np.rot90(abs(stamp_a.data))))*3.631e-6
    AB_mag = -2.5*np.log10(Jy/3631)

    plt.clf()
    plt.imshow(AB_mag)
    plt.title(np.min(AB_mag))
    plt.colorbar()
    plt.savefig('imaging/SDSS_AB_mag_'+str(sdss)+'.png')

    

    print('mags', np.min(AB_mag))
    
    
    
  #this is the r-band image, and we divide by the conversion factor for MaNGA, 0.005
    #We want things to be in units of counts
    #This is the average conversion between nmgy and counts for SDSS, but Source Extractor and Galfit only
        #require a rough amount
    
    
    #STOP
    
    #camera_data_ivar=abs(im[5].data)
    #calculate the percent error to then convert to counts:
    
    #percent_e=abs((1/camera_data_ivar**2)/im[4].data)
    camera_data_sigma = np.sqrt(abs(camera_data))
    
    #psf=im[6].data

    '''
    The first step is creating some .fits files for use by galfit and Source extractor
    '''
    
    
    
    outfile = 'imaging/pet_radius_'+str(sdss)+'.fits'
    hdu = fits.PrimaryHDU(abs(camera_data))#was d[0] (the unconvolved version)                                                                                      
    hdu_number = 0
    hdu.writeto(outfile, overwrite=True)
    hdr=fits.getheader(outfile, hdu_number)

    hdr['EXPTIME'] = 1
    hdr['EXPTIME']

    hdu.writeto(outfile, overwrite=True)

    
    

     
    
    '''Make a psf file for Galfit'''
    #outfile = 'imaging/psf_'+sdss+'.fits'
    #hdu = fits.PrimaryHDU(psf)
    #hdu_number = 0
    #hdu.writeto(outfile, overwrite=True)
    
    
  
    outfile = 'imaging/out_convolved_'+str(sdss)+'.fits'
    hdu = fits.PrimaryHDU((camera_data))
    hdu_number = 0
    hdu.writeto(outfile, overwrite=True)
    hdr=fits.getheader(outfile, hdu_number)
    
    hdr['EXPTIME'] = 1
    hdr['EXPTIME']
    
    hdu.writeto(outfile, overwrite=True)

    outfile = 'imaging/out_sigma_convolved_'+str(sdss)+'.fits'
    hdu = fits.PrimaryHDU((camera_data_sigma))
    hdu_number = 0
    hdu.writeto(outfile, overwrite=True)
    hdr=fits.getheader(outfile, hdu_number)
    
    hdr['EXPTIME'] = 1
    hdr['EXPTIME']
    hdu.writeto(outfile, overwrite=True)


 
    pixelscale =  0.5#This is universally 0.5" spaxels for MaNGA                                                                                                    
    redshift = df['redshift'][i]
    kpc_arcmin=cosmo.kpc_proper_per_arcmin(redshift)#insert the redshift                                                                                            
    '''Divide the pixelscale (kpc) by kpc/arcsec to get arcsec                                                                                                      
    size of pixels'''
    kpc_pix=pixelscale/(kpc_arcmin.value/60)#This is the number of kpc/pixel
    '''Running sextractor before galfit to determine good guess input parameters'''
    

    write_sex_default(str(sdss))
    run_sex(str(sdss))
    sex_out=sex_params_galfit(str(sdss))
    #output if there are 2 bright sources aka 2 stellar bulges:
        #return sex_pet_r, minor, flux, x_max, y_max, n_bulges, eff_radius_1, B_A_1, PA_1, back_1, x_max_2, y_max_2, mag_max_2, eff_radius_2, B_A_2, PA_2
        #

    #output if 1 bulge:
        #return sex_pet_r, minor, flux, x_max, y_max, n_bulges, eff_radius_1, B_A_1, PA_1, back_1
    
    bg_data=plot_sex(str(sdss), sex_out[3], sex_out[4])
    length_gal=np.shape(camera_data)[0]
    
    
    

    '''now put together all of the necessary Galfit inputs - it is a little finnicky and needs to know
    magnitude guesses, which is annoying because everything is in counts and often there's no exposure time
    provided, but you can guess the magnitude as below by setting the mag_zpt and then calculating the flux ratio
    given the source extractor background and main source counts'''

    
 

    '''This is scaled up using the background flux and the flux from the petrosian output'''
    
    '''Now use the output of source extractor to also determine the petrosian radius,
    the semimajor and semiminor axes, the flux guess, the bg level, the PA, and the axis ratio'''
    r_sex=sex_out[0]#petrosian radius (aka semimajor axis because it is an elliptical aperture)
    b_sex=sex_out[1]#minor axis of the above
    flux_sex=sex_out[2]#flux of the brightest aperture in counts
    x_pos_1=sex_out[3]
    y_pos_1=sex_out[4]
    num_bulges=sex_out[5]
    eff_rad_1=sex_out[6]#effective radius
    AR_1=sex_out[7]#ellipticity
    PA1=sex_out[8]#position angle on sky
    bg_level=sex_out[9]#bg counts

    
 
    Area=(b_sex*r_sex*np.pi)
    mag_flux = -2.5*np.log10(((flux_sex)*0.005*3.631*10**(-6))/3631)
    mag_flux_Area = -2.5*np.log10(((flux_sex/Area)*0.005*3.631*10**(-6))/3631)
    print('in mags', mag_flux_Area, mag_flux)
    
 
    
    #Define an arbitrary magnitude zeropoint for Galfit
    mag_zpt=26.563

    #try setting mag_zpt as the background magnitude and scale up from there with the flux
    
    mag_guess=(-2.5*math.log10((flux_sex)/bg_level)+mag_zpt)


    if num_bulges==2:
        #further define some things from sourcextractor output:
        x_pos_2=sex_out[10]
        y_pos_2=sex_out[11]
        flux_sex_2=sex_out[12]
        eff_rad_2=sex_out[13]
        AR_2=sex_out[14]
        PA2=sex_out[15]
        
        
        mag_guess=(-2.5*math.log10((flux_sex)/bg_level)+mag_zpt)

        mag_guess_2=(-2.5*math.log10((flux_sex_2)/bg_level)+mag_zpt)

        #Prepares the galfit feedme file:
        f=write_galfit_feedme(sdss, x_pos_1, y_pos_1, x_pos_2, y_pos_2, mag_guess, mag_zpt, num_bulges, length_gal, eff_rad_1, eff_rad_2, mag_guess_2, AR_1, PA1, AR_2, PA2, bg_level)
        
    else:
        
        f=write_galfit_feedme(sdss, x_pos_1, y_pos_1, 0,0, mag_guess, mag_zpt, num_bulges, length_gal, eff_rad_1,0,0, AR_1, PA1, 0, 0, bg_level)
        

    '''Runs galfit inline and creates an output file'''
    g=run_galfit(str(sdss))

    output='imaging/out_'+str(sdss)+'.fits'
    try:
        #This is the output result fit from Galfit:
        out=pyfits.open(output)
    except FileNotFoundError:
        #sometimes Galfit fails and you need to move on with your life.
        #It should really work for the majority of cases though.
        print('Galfit failed', sdss)
        continue
    
    '''This extracts useful parameters from Galfit'''
    h=galfit_params(sdss,num_bulges,kpc_pix)
    if h[2]==999:
        continue


    #Re-adjust so the important parameters are now Galfit outputs:
    sep=h[0]
    flux_r=h[1]
    ser=h[2]# you can use many of these predictors to check the output, but the only important one is the sersic profile
    inc=h[3]


    '''plt.clf()
    ax = plt.subplot(111,aspect='equal')
    ax.imshow(camera_data)
    from matplotlib.patches import Ellipse
    ellipse=Ellipse((x_pos_1, y_pos_1),2*r_sex, 2*b_sex, PA1,ec='white', fc="none", lw=3)
    ax.add_artist(ellipse)
    plt.savefig('../MaNGA_Papers/Paper_I/troubshoot_ellipse_'+str(sdss)+'.pdf')
    STOP'''

    limiting_sb=input_petrosian_radius(r_sex, b_sex, flux_sex)
    print('limiting sb at petrosian', limiting_sb)

    threshold = np.array(limiting_sb*np.ones((np.shape(camera_data)[0],np.shape(camera_data)[1])))
        #print(threshold)
    npixels = 1  # was 1 minimum number of connected pixels
        #threshold=0.1

    

    segm = photutils.detect_sources(camera_data, threshold, npixels)
        
 
    label = np.argmax(segm.areas) + 1
    segmap = segm.data == label

 
    segmap_float = ndimage.uniform_filter(np.float64(segmap), size=10)

    new_threshold = (0.75*(np.max(segmap_float)-np.min(segmap_float))+np.min(segmap_float))
    segm = photutils.detect_sources(segmap_float, new_threshold, npixels, connectivity=8)

    label = np.argmax(segm.areas) + 1
    segmap = segm.data == label

          
    segmap_float = ndimage.uniform_filter(np.float64(segmap), size=10)
    segmap = segmap_float >  0.5
    #else:
        #segmap = segmap_float >  0.5


 
    
    

    threshold = photutils.detect_threshold(camera_data, snr=1.5)#, background=0.0)#was 1.5
    npixels = 5  # minimum number of connected pixels was 5
    segm = photutils.detect_sources(camera_data, threshold, npixels)

    label = np.argmax(segm.areas) + 1
    segmap = segm.data == label
    import scipy.ndimage as ndi

    segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
    segmap = segmap_float > 0.5

    
    #Now we need to do some thresholding using the limiting
    #surface brightness at the petrosian radius:
    #threshold = np.array(limiting_sb*np.ones((np.shape(camera_data)[0],np.shape(camera_data)[1])))
    #masked=ma.masked_where(camera_data < limiting_sb, camera_data)
    #plt.clf()
    #plt.imshow(masked)
    #plt.savefig('sdss/masked.pdf')
    #npixels = 1  # minimum number of connected pixels                                                                                                               
                                                                                                                                                      

    
    

    #segmap_float = ndimage.uniform_filter(np.float64(segmap), size=10)
    #segmap = segmap_float > (0.5*(np.max(segmap_float)-np.min(segmap_float))+np.min(segmap_float))
    #print('new thresh', 0.5*(np.max(segmap_float)-np.min(segmap_float))+np.min(segmap_float))
    

    '''If its all black then select from before'''
    
    '''if np.min(segmap_float) > 0.5:
        print('getting a better segmap')
        segmap=np.ones((np.shape(camera_data)[0],np.shape(camera_data)[1]))'''
    
    plt.clf()
    #plt.title(str(0.5*(np.max(segmap_float)-np.min(segmap_float))+np.min(segmap_float)))
    fig=plt.figure()
    ax1=fig.add_subplot(141)
    im1=ax1.imshow(camera_data, norm=matplotlib.colors.LogNorm())
    plt.colorbar(im1)
    
    ax2=fig.add_subplot(142)
    im2=ax2.imshow(segm)
    plt.colorbar(im2)
    
    #ax3=fig.add_subplot(143)
    #im3=ax3.imshow(segmap_float)
    #plt.colorbar(im3)#this needs to be pretty much fully 0 to 1, when its half that were in trouble
    
    ax4=fig.add_subplot(144)
    im4=ax4.imshow(segmap)
 #   plt.colorbar(im2)
    plt.savefig('sdss/sigma_probs.pdf')


    '''Vicente says to do some background estimation'''
    from astropy.stats import SigmaClip
    from photutils import Background2D, MedianBackground
    sigma_clip = SigmaClip(sigma=3., iters=10)
    bkg_estimator = MedianBackground()
    try:
        bkg = Background2D(camera_data, (np.shape(camera_data)[0],np.shape(camera_data)[1]),edge_method='pad',
                           filter_size=(3,3),sigma_clip=sigma_clip,bkg_estimator=bkg_estimator)
    except ValueError:
        continue

    
    
    '''plt.clf()
    fig=plt.figure()
    ax1=fig.add_subplot(211)
    im1=ax1.imshow(camera_data)
    plt.colorbar(im1)
    
    ax2=fig.add_subplot(212)
    im2=ax2.imshow(camera_data_sigma)
    plt.colorbar(im2)
    plt.savefig('diagnostic_sig.pdf')
    STOP'''

    #call statmorph
    source_morphs = statmorph.source_morphology(camera_data, segmap, 
                                                weightmap=camera_data_sigma,skybox_size=20)#,weightmap=camera_data_sigma)#psf=psf,
    try:
        morph=source_morphs[0]
    except IndexError:
        print('STATMORPH FAIL')
        continue
    
    
    morph=source_morphs[0]
    
    if morph.sn_per_pixel < 2.5:
        continue
    #if morph.flag==1:
    #    '''This means the measurement is unreliable'''
    #    continue
    print('S_N', morph.sn_per_pixel, 'mag', mag_flux)

    plt.clf()
    fig = make_figure(morph)
    fig.savefig('diagnostics_SDSS_seg_'+str(sdss)+'.png')

    S_N_list.append(morph.sn_per_pixel)
    mag_list.append(mag_flux)

    
    
    
    
    print('Gini =', morph.gini)
    print('M20 =', morph.m20)
    print('F(G, M20) =', morph.gini_m20_bulge)
    print('sn_per_pixel =', morph.sn_per_pixel)
    print('C =', morph.concentration)
    print('A =', morph.asymmetry)
    print('S =', morph.smoothness)
    print('sersic_n =', morph.sersic_n)
    print('sersic_rhalf =', morph.sersic_rhalf)
    print('sersic_ellip =', morph.sersic_ellip)
    print('sersic_theta =', morph.sersic_theta)
    print('flag =', morph.flag)
    print('flag_sersic =', morph.flag_sersic)
    
    
    gini=morph.gini
    m20=morph.m20
    con=morph.concentration
    asy=morph.asymmetry
    clu=morph.smoothness
    
    
    '''Now we need to apply the segmentation mask to the normal surface brightness'''
    
    n = img_assy_shape(segmap,sdss)
 

    print('my assy', n, 'shape assy morhp', morph.shape_asymmetry)
    
    n = morph.shape_asymmetry
    if ser > 20:
        ser =morph.sersic_n
    
    if counter==0:
        file.write('Counter'+'\t'+
            'ID'+'\t'+'Merger?'+'\t'+'# Bulges'+'\t'+'Sep'+'\t'+'Flux Ratio'+'\t'+'Gini'+'\t'+'M20'+'\t'+'C'+'\t'+'A'+'\t'+'S'+
                    '\t'+'Sersic n'+'\t'+'A_s'+'\n')#was str(np.shape(vel_dist)[1]-1-j)

    #The Merger? column is meaningless; MergerMonger will classify while ignoring these columns
    #but it is important to leave it in for ease so that the dataframes have the same headers and match formats
    file.write(str(counter)+'\t'+str(sdss)+'\t'+str(merger[i])+'\t'+str(num_bulges)+'\t'+str(sep)+'\t'+str(flux_r)+
                '\t'+str(gini)+'\t'+str(m20)+'\t'+str(con)+'\t'+str(asy)+'\t'+str(clu)+'\t'+str(ser)+'\t'+str(n)+'\n')#was str(np.shape(vel_dist)[1]-1-j)
    
    counter +=1
    kept_ids.append(sdss)

    #To save space, I'm deleting the preimaging MaNGA files after their imaging predictors are extracted:
    
    #os.system('rm sdss/preimage-'+str(mangaid)+'.fits')
    #os.system('rm preim/preimage-'+str(mangaid)+'.fits.gz')

    #I also delete the Galfit and source extractor related .fits files because they are big:
    
    os.system('rm imaging/pet_radius_'+str(sdss)+'.fits')
    #os.system('rm imaging/out_'+str(manga_list[i])+'.fits')
    os.system('rm imaging/out_sigma_convolved_'+str(sdss)+'.fits')
    os.system('rm imaging/out_convolved_'+str(sdss)+'.fits')
    os.system('rm imaging/galfit.feedme_'+str(sdss))

    #if counter==100:#
#        break
    
    
file.close()

print(S_N_list, np.mean(S_N_list))
print(mag_list, np.mean(mag_list))

from numpy.polynomial.polynomial import polyfit

b, m = polyfit(S_N_list, mag_list, 1)

plt.clf()
plt.scatter(S_N_list, mag_list)
plt.plot(S_N_list, np.array(S_N_list) * m + b)
plt.xlabel('S/N')
plt.ylabel('App Mag')
plt.tight_layout()
plt.savefig('S_N_mag_SDSS.png')
print('Finished and Congrats!')

print('solution', 2.5*m+b)



    
