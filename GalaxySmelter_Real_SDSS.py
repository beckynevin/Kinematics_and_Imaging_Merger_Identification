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

    file.write('VERBOSE_TYPE     FULL         # can be QUIET, NORMAL or FULL'+'\n')
    file.write('WRITE_XML        Y              # Write XML file (Y/N)?'+'\n')
    file.write('XML_NAME         imaging/sex.xml        # Filename for XML output'+'\n')


    file.close()

'''Runs source extractor from within python'''
    
def run_sex(name):
    
    os.system("sex -c imaging/default_"+name+".sex "+"imaging/pet_radius_"+name+".fits")
    

'''Extracts parameters from source extractor that are useful as a Galfit input/ first guess'''
def sex_params_galfit(name):
    
    
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
def run_galfit(name):
    os.system("galfit imaging/galfit.feedme_"+str(name))
    
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



 
'''
Now we begin.

Here, as an example, I feed the code a list of galaxies in MaNGA identified by Fu et al. 2018 as close pairs.
If you are a MaNGA team member with access to this data, you can run these galaxies, otherwise, use another set.
'''

#SDSS superclean merger Galaxy Zoo
sdss_list=[587724197207212176, 587731512073650262, 587738195036799112, 587732156315402630, 587728669878845746, 587737825675903236, 587732578831434101, 588013383261880542, 587741386202218597, 587741386202218596, 587732469850767564, 587732469850767563, 587731520668434686, 588023240205795356, 587725469597499626, 588013382191349914, 588013382191349913, 587741817313558731, 587734621635149881, 587731521207992450, 587728930808135685, 587725818015842502, 587741532774793359, 587738196660781221, 587738617285836937, 587728881411424336, 587734950200803557, 587734950200803558, 588016528244670522, 587745539980787848, 587738618094354456, 587731500260655172, 587735662090125418, 587739647816761430, 588017703997145160, 588848899905552544, 588017720638767190, 587731870705385675, 587741603100164157, 587731890575835356, 587742774017786012, 587739408388063264, 587739646747017350, 587725039015297179, 587741533324640327, 588017726007541823, 588017726007541822, 587735695904997435, 588017112367890517, 587739507151863911, 587731891114672276, 587724650867130596, 587739504467902597, 588013383816904792, 587732484359716927, 587739097525714956, 587732769981988969, 587739304753758319, 587739304753758323, 588017702395904123, 587742865288659012, 587739720287584329, 587739305291677774, 587739305291677773, 588017721183240317, 587742772952170547, 588017730301198375, 587741602571223189, 587739721900163101, 587741721212092511, 587738569780428942, 588298664650145819, 588017112372936752, 587739505010802837, 588017111299522645, 587726031182299335, 587742061077987425, 588017605225545758, 587739407324479578, 588018057256239110, 587741602038612021, 587735665840881790, 587738568712192204, 588011219672170717, 588011218062016644, 587742592547160124, 587735430694043762, 588017726021369993, 588017606302498929, 587739809948237949, 587742594696151179, 587736477588848682, 587726033873272951, 587726033873272950, 587739380448952409, 588848899927441439, 587736586574758004, 587735666380111970, 587739828738195570, 587739829275721914, 587742014907547887, 587742014907547886, 587742629059362936, 588017979978350823, 587736586040443080, 587739458299756738, 587742627451699301, 587742013300146345, 587726102025797805, 587736975808266400, 588018091075895558, 588018055654998067, 587742013303619894, 587742013303619895, 587736975809708043, 587739827672908020, 587739720846606568, 588018254831812781, 588018091081138197, 587725818037862416, 587739845397446904, 588018090012311863, 588007004735668429, 587725505019117816, 587725576426356739, 587726879953518907, 587726878885478656, 587727222484762718]
ra_list=['01:30:37.76', '01:58:16.35', '07:34:03.45', '07:57:33.42', '08:04:22.68', '08:06:55.01', '08:20:11.59', '08:29:17.37', '08:30:31.36', '08:30:31.68', '08:38:51.95', '08:38:52.36', '08:50:57.24', '08:53:12.35', '09:00:23.64', '09:00:25.37', '09:00:25.61', '09:05:37.08', '09:15:02.50', '09:15:55.53', '09:22:29.44', '09:29:04.24', '09:36:33.94', '09:46:43.44', '09:51:12.52', '09:52:13.28', '09:53:49.00', '09:53:49.47', '09:55:59.89', '09:57:39.84', '10:22:56.55', '10:36:42.75', '10:44:28.48', '10:47:11.19', '10:53:32.06', '10:54:47.31', '11:03:41.37', '11:20:10.20', '11:21:31.62', '11:21:50.68', '11:22:03.68', '11:26:48.65', '11:30:15.64', '11:33:59.46', '11:46:26.15', '11:47:19.76', '11:47:19.95', '11:48:39.32', '11:51:23.05', '11:53:27.30', '11:53:35.98', '11:55:34.23', '11:58:09.91', '12:04:39.55', '12:10:41.77', '12:11:56.28', '12:12:13.56', '12:17:49.32', '12:17:49.71', '12:21:24.20', '12:22:10.20', '12:25:12.28', '12:29:40.60', '12:29:41.61', '12:36:20.10', '12:39:02.26', '12:40:23.33', '12:43:35.50', '12:46:10.11', '12:48:46.85', '12:49:55.54', '12:56:48.28', '13:01:18.36', '13:03:08.20', '13:04:39.23', '13:08:38.07', '13:09:37.56', '13:15:17.26', '13:20:35.41', '13:23:17.97', '13:26:51.76', '13:41:39.23', '13:41:40.89', '13:42:20.35', '13:46:36.35', '13:48:14.33', '13:51:59.18', '13:54:19.69', '13:56:38.56', '13:56:53.25', '14:03:19.25', '14:08:49.47', '14:09:02.65', '14:09:02.77', '14:13:58.54', '14:14:47.15', '14:16:31.80', '14:16:51.56', '14:17:24.07', '14:23:55.51', '14:27:22.83', '14:27:23.13', '14:28:18.88', '14:33:31.00', '14:43:29.85', '14:52:29.05', '14:54:32.02', '14:56:08.54', '15:04:46.55', '15:16:16.04', '15:17:15.23', '15:18:06.13', '15:28:07.16', '15:28:07.47', '15:29:29.90', '15:35:54.86', '16:02:29.03', '16:02:33.14', '16:05:22.32', '16:11:40.82', '16:38:15.38', '16:41:40.86', '17:01:11.64', '17:05:06.84', '17:28:23.83', '21:49:48.65', '22:42:01.75', '23:36:12.85']
dec_list=['+13:12:52.0', '-00:31:18.8', '+43:32:41.3', '+25:34:36.5', '+40:38:55.9', '+50:03:00.6', '+04:22:08.2', '+35:04:45.6', '+18:12:07.5', '+18:12:22.0', '+33:12:31.4', '+33:12:26.7', '+40:11:34.5', '+16:26:19.5', '+53:46:19.3', '+39:03:53.7', '+39:03:49.3', '+17:42:50.2', '+36:05:23.7', '+44:19:58.0', '+50:25:48.5', '+61:06:59.0', '+23:26:38.8', '+68:50:02.6', '+31:35:38.9', '+04:53:44.2', '+10:58:27.1', '+10:58:32.7', '+39:54:46.9', '+14:15:43.4', '+34:46:46.9', '+54:47:35.6', '+40:00:54.7', '+30:43:27.8', '+12:03:43.0', '-00:12:27.7', '+40:36:43.5', '+55:13:45.1', '+27:41:17.4', '+56:32:03.3', '+16:40:25.0', '+35:14:54.2', '+31:19:31.0', '-03:36:23.0', '+30:05:23.4', '+07:52:29.3', '+07:52:43.1', '+56:56:39.1', '+48:09:36.2', '+33:30:55.6', '+57:37:21.3', '-01:21:16.3', '+31:26:29.3', '+52:57:26.3', '+50:48:41.7', '+37:21:09.9', '+08:48:56.7', '+35:44:50.4', '+35:44:50.4', '+11:27:03.9', '+18:16:04.0', '+29:39:22.1', '+36:11:57.9', '+36:11:53.1', '+42:35:28.3', '+16:18:24.7', '+08:10:23.0', '+27:53:20.3', '+30:43:54.9', '+26:44:55.0', '+15:16:39.5', '+48:17:43.9', '+48:03:30.5', '+31:32:44.0', '+47:07:20.1', '+01:16:12.5', '+21:47:06.8', '+44:24:25.6', '+34:08:21.6', '+52:39:06.6', '+26:35:27.4', '+55:40:14.0', '+13:30:14.6', '+61:48:38.8', '+60:18:46.8', '+15:25:38.2', '+48:07:29.0', '+07:23:12.4', '+43:35:08.8', '+24:58:02.7', '+16:38:05.8', '+10:55:49.8', '+03:11:29.6', '+03:11:39.3', '+30:57:19.4', '-00:00:13.2', '+39:35:20.6', '+54:10:47.4', '+24:53:55.1', '+24:58:41.1', '+20:05:49.4', '+20:05:47.9', '+15:24:58.9', '+35:55:58.9', '+37:12:14.3', '+26:29:30.4', '+13:02:06.5', '+17:18:34.4', '+04:22:27.1', '+28:27:00.2', '+38:36:25.9', '+42:44:45.1', '+15:18:03.7', '+15:18:09.2', '+27:12:05.4', '+18:26:44.4', '+18:05:55.2', '+37:34:52.9', '+31:56:22.9', '+52:27:26.9', '+11:34:12.5', '+23:58:59.6', '+35:51:14.4', '+62:51:42.9', '+57:32:43.4', '-06:53:46.1', '-08:46:27.6', '+15:00:24.5']

#SDS superclean ellipticals
#sdss_list=[587730775499407394, 587731187277758607, 588015510343188562, 587727226764001335, 587727226764001349, 587730774962602216, 587731186740953163, 588015509806448799, 587730774425862345, 587730774962733276, 588015509269577819, 587731187277889699, 588015509806514333, 588015509806514377, 587727226227392604, 587731187814891689, 588015510343516313, 587727223024582862, 587730773889187972, 588015510343647360, 587727221950906570, 588015510343647386, 587731186204475615, 587727220877295779, 587727180597624973, 588015509806907572, 587730773889450117, 587727177913270351, 587727227838267555, 587727226227785819, 587731186741608591, 587727223024910513, 588015510343909538, 587730773889581202, 588015508733362219, 587731186204868769, 588015509270364333, 587727221414494464, 587727221951430780, 587727225691177054, 587727225691177077, 587731185131258008, 587727180061147206, 587727225691177110, 587731187815678052, 587727221414625329, 587727220877754576, 587730774426779780, 587727177913794628, 587727178987536449, 588015507660013699, 588015508733820990, 587727227302117474, 587727178987733143, 588015508734017670, 587731187816136707, 588015509270954123, 587731186205524102, 587731185131782170, 587727226228572258, 588290882175697039, 587730773890433205, 588290882175762627, 587727180598673479, 587727227302510691, 587731187279462454, 587730774427435164, 587731186205786263, 587731187279593490, 587727226765770881, 587727225692094555, 587727227839578234, 587727227302707352, 588015508197671026, 587727179525259420, 587727225155289200, 587727227302838422, 588290882176155793, 587727180599066724, 587730773890826309, 587727180599066732, 588290882176155831, 587727180599132320, 587731186206179425, 588015509271674958, 587731185132503151, 588290880565739627, 587731185132568693, 587727178451845246, 587727225692487756, 588015508198064315, 587730773354283087, 587727180599394441, 588290881639743516, 588015508198260810, 587727225692749926, 587730773354479638, 588015509272199310, 588015508735328323, 587727180062851195, 587730772817739967, 587727180062916662, 588290881639940281, 587727178452303992, 588015509272330337, 587730774965354640, 587727179526111364, 587731185670029367, 588015509809201262, 588015510346137716, 587727225693143097, 587724233710239870, 587724198274400379, 587731186743967753, 587731186743967777, 587731185133355176, 588015508735721577, 587731185670291527, 588015508735721609, 588015509809463435, 588015507662045325, 587727180063309931, 587724197737660527, 587724198811402376, 587724198274597028, 587724234247372908, 587727225156665355, 587731186744230034, 587724232099954700, 587731185133617279, 588015508735983718, 587724197737922625, 587727178452959334, 587724199885537310, 587724198811861121, 588015508736311392, 587727178453221380, 587727178453221476, 587724199348797600, 587727180600836147, 587727225157124201, 587727225157124149, 587727178453418135, 587724232100479089, 587727227304738903, 588015509273509984, 587724233711157377, 587724197738512497, 587727227304870000, 587727225694257237, 587724234248159295, 587724198275449013, 588015508199899287, 588015508199899299, 588015509273641038, 587724198275514500, 587731186208342149, 587727178453811296, 587731187282084017, 587727227842003062, 587731187282149525, 588015507663224976, 587727180064555103, 587727227305197792, 587727178453942299, 587727227305263285, 587727180064620635, 587727227305263139, 588015509810839636, 587727225157779564, 587727180064620675, 587724198275776699, 587727227842199689, 587727179527815279, 587727180064686217, 587727227305328828, 588015510347776169, 588015508737163423, 587727227305394267, 587727227305394310, 587724197739036855, 587724233174876337, 587727178454138922, 587724233174876394, 587724233175072929, 587724199886717111, 587731185135124605, 588015510348169354, 588015509274427469, 588015509274493041, 588015508200751274, 587724199350042803, 587727178991403062, 588015510348300342, 587724199350108310, 587731187282870448, 587731185135386760, 587727179528404997, 587727179528405177, 587727178991534229, 587731186746130595, 587731186746196163, 587731187283067026, 587727227842986096, 588015510348562572, 587724233712468072, 587727180602343544, 588015509811691722, 588015507664207949, 587727177917988936, 587724234249404575, 587727180602474580, 587727180602540112, 587724198276825213, 587727226769506398, 587724198276890820, 588015507664470176, 587731187283394709, 587731186746589190, 588015508201472118, 588015508201472181, 587727178992123934, 588015508201472210, 588015509275279473, 587724199887700060, 587727180065931425, 587727180065996868, 587727177918513282, 588015508738539628, 588015510349152365, 587727177918578736, 587727226232963252, 587727178992320624, 587727226233028742, 587731185136304258, 587731185136304268, 588015507664928923, 587731185136304288, 588015507664928792, 587727226769965185, 587724231565705291, 588015509275541670, 588015508201799826, 588015507664994403, 587724199351156966, 587731185673371705, 588015509812543622, 587727226770096309, 587724198814351534, 587724198814417027, 588015508202061873, 588015509812674728, 587727227843969057, 588015509812674645, 587727177918971996, 587727177918972029, 587727180603392098, 587731513409405093, 587724197204000957, 587731512067424361, 587727179529846915, 587724198814810134, 587727178993107062, 587731513678168196, 587724234250715257, 588015507665649689, 587724234250780783, 588015507665715341, 587727180603850759, 588015508202651799, 587731511530815625, 587731511530815630, 587731512604622918, 588015508739588187, 587724199888748690, 587724199888814227, 587727179530240094, 587724197204525250, 587727226770882649, 587727180603981888, 587727226234077265, 587727227844690017, 587727178456563794, 588015509813592105, 588015510350463162, 587724197204721900, 587727226771144769, 587731513678626974, 587724199352205454, 587731513678692455, 588015509276721307, 587724232103755886, 587724199889076343, 587724232640626695, 588015510350528621, 587731513141821482, 588015508739915896, 587724233177497691, 587727180067438686, 587731512605016156, 587731514215694430]
#ra_list=['00:00:07.62', '00:00:44.15', '00:00:51.56', '00:01:00.19', '00:01:05.80', '00:01:07.75', '00:01:34.63', '00:01:49.57', '00:01:53.36', '00:01:57.95', '00:02:02.81', '00:02:03.61', '00:02:36.20', '00:02:48.88', '00:03:21.63', '00:03:22.49', '00:03:45.47', '00:03:52.03', '00:04:02.04', '00:04:48.94', '00:04:53.38', '00:04:58.59', '00:05:12.07', '00:06:05.73', '00:06:07.05', '00:06:08.55', '00:06:16.19', '00:06:18.00', '00:06:22.17', '00:07:09.16', '00:07:15.64', '00:07:17.68', '00:07:27.00', '00:07:40.42', '00:07:55.11', '00:08:56.10', '00:09:10.47', '00:09:18.54', '00:09:26.77', '00:09:39.67', '00:09:47.97', '00:09:50.47', '00:09:52.94', '00:09:58.47', '00:10:15.63', '00:10:17.16', '00:10:30.13', '00:10:39.16', '00:10:49.70', '00:11:07.72', '00:11:31.35', '00:12:06.09', '00:12:37.91', '00:13:05.54', '00:13:52.45', '00:14:26.23', '00:14:31.89', '00:14:34.70', '00:14:39.67', '00:14:42.81', '00:15:04.13', '00:15:36.98', '00:15:55.69', '00:16:12.73', '00:16:17.64', '00:16:47.01', '00:16:58.28', '00:17:09.89', '00:17:34.47', '00:17:44.99', '00:18:11.94', '00:18:14.95', '00:18:19.60', '00:18:32.54', '00:18:58.45', '00:19:02.44', '00:19:28.08', '00:19:34.56', '00:19:38.77', '00:19:38.79', '00:19:40.24', '00:19:44.30', '00:20:17.07', '00:20:32.95', '00:20:48.94', '00:21:07.62', '00:21:10.75', '00:21:42.38', '00:21:57.54', '00:22:02.99', '00:22:29.38', '00:22:49.24', '00:22:53.29', '00:23:47.86', '00:23:54.20', '00:24:01.05', '00:24:16.20', '00:25:37.70', '00:25:39.13', '00:25:41.77', '00:25:58.89', '00:26:01.77', '00:26:01.98', '00:26:33.61', '00:26:49.05', '00:26:55.44', '00:26:58.10', '00:27:05.61', '00:27:14.99', '00:27:32.17', '00:27:46.21', '00:28:40.33', '00:28:44.83', '00:28:53.40', '00:29:01.48', '00:29:14.20', '00:29:28.42', '00:29:32.41', '00:29:35.59', '00:29:47.15', '00:29:49.35', '00:29:58.80', '00:30:01.33', '00:30:13.12', '00:31:00.14', '00:31:27.42', '00:31:36.56', '00:31:37.81', '00:31:43.05', '00:31:44.85', '00:31:48.96', '00:32:41.06', '00:32:41.19', '00:33:42.43', '00:34:17.33', '00:34:39.35', '00:34:40.08', '00:34:59.82', '00:35:18.26', '00:35:43.52', '00:35:58.89', '00:35:59.88', '00:37:00.39', '00:37:08.03', '00:37:15.66', '00:37:35.66', '00:37:38.09', '00:37:55.67', '00:38:02.62', '00:38:10.47', '00:38:51.51', '00:38:54.22', '00:39:12.49', '00:39:14.92', '00:39:21.82', '00:39:36.21', '00:40:14.72', '00:40:22.16', '00:40:33.46', '00:40:46.48', '00:40:52.43', '00:41:03.01', '00:41:18.40', '00:41:33.41', '00:41:36.09', '00:41:48.23', '00:41:50.17', '00:41:50.47', '00:41:52.17', '00:41:53.30', '00:42:03.27', '00:42:12.93', '00:42:27.71', '00:42:33.99', '00:42:39.54', '00:42:41.97', '00:42:45.39', '00:42:49.51', '00:42:54.69', '00:43:00.63', '00:43:10.19', '00:43:11.82', '00:43:12.85', '00:43:25.23', '00:45:01.06', '00:45:16.28', '00:45:22.17', '00:46:18.43', '00:46:24.20', '00:46:44.85', '00:47:05.26', '00:47:05.64', '00:47:08.78', '00:47:16.00', '00:47:28.63', '00:47:34.74', '00:47:40.95', '00:47:54.12', '00:48:21.97', '00:48:30.46', '00:48:51.82', '00:49:32.47', '00:49:42.42', '00:49:46.73', '00:49:48.53', '00:49:49.90', '00:49:52.68', '00:49:57.82', '00:50:00.45', '00:50:17.57', '00:50:49.14', '00:50:56.05', '00:51:43.11', '00:51:57.28', '00:52:07.26', '00:52:21.36', '00:52:22.20', '00:52:30.43', '00:52:49.80', '00:53:12.56', '00:53:29.39', '00:53:36.34', '00:53:42.39', '00:54:07.17', '00:54:22.25', '00:54:28.47', '00:54:44.99', '00:54:58.94', '00:55:07.05', '00:55:07.25', '00:55:25.72', '00:55:32.45', '00:55:42.70', '00:55:51.89', '00:56:06.29', '00:56:08.69', '00:56:10.67', '00:56:17.02', '00:56:18.15', '00:56:21.88', '00:56:23.01', '00:56:26.31', '00:56:27.43', '00:56:52.71', '00:57:40.31', '00:57:46.47', '00:57:53.27', '00:57:56.52', '00:58:14.41', '00:58:45.18', '00:58:46.78', '00:58:53.16', '00:58:53.16', '00:59:01.19', '00:59:06.88', '00:59:30.08', '00:59:39.22', '00:59:54.91', '01:00:10.86', '01:01:26.16', '01:01:31.34', '01:02:18.60', '01:02:43.17', '01:02:43.37', '01:02:57.91', '01:03:01.73', '01:03:40.00', '01:03:44.91', '01:03:47.82', '01:04:01.74', '01:04:04.05', '01:04:04.52', '01:04:25.28', '01:04:33.51', '01:04:35.80', '01:04:51.24', '01:05:07.85', '01:05:08.38', '01:05:08.38', '01:05:22.39', '01:05:24.29', '01:05:34.80', '01:05:48.55', '01:06:55.79', '01:07:01.39', '01:07:06.47', '01:07:15.70', '01:07:15.95', '01:07:16.04', '01:07:21.93', '01:07:26.81', '01:07:30.48', '01:07:32.65', '01:07:37.01', '01:07:40.54', '01:07:53.67', '01:07:55.75', '01:07:57.71', '01:08:00.11', '01:08:17.14', '01:08:31.67']
#dec_list=['+15:50:03.2', '+00:40:22.8', '+00:53:23.9', '-09:36:12.4', '-09:42:40.8', '+15:28:10.1', '+00:16:28.0', '+00:34:01.0', '+15:02:02.4', '+15:28:21.2', '+00:04:36.7', '+00:37:54.7', '+00:25:16.3', '+00:28:34.2', '-09:58:59.8', '+01:03:28.4', '+01:02:41.0', '+15:46:32.4', '+14:40:05.2', '+00:57:04.1', '+14:55:58.6', '+00:57:43.5', '-00:09:02.6', '+13:58:43.8', '-08:56:22.2', '+00:27:20.0', '+14:34:38.6', '-11:11:47.4', '-08:55:51.0', '-09:59:10.1', '+00:22:25.2', '+15:48:55.5', '+00:55:12.5', '+14:45:06.7', '-00:13:41.1', '-00:11:38.1', '+00:05:51.0', '+14:23:30.7', '+14:50:33.9', '-10:30:56.4', '-10:28:36.5', '-00:50:57.4', '-09:24:17.3', '-10:36:59.9', '+01:10:09.9', '+14:22:33.7', '+14:01:30.4', '+15:06:15.3', '-11:08:13.0', '-10:16:36.6', '-01:14:23.8', '-00:24:54.8', '-09:12:40.4', '-10:22:38.5', '-00:23:48.2', '+01:03:50.9', '+00:00:29.8', '-00:00:45.0', '-00:52:28.8', '-10:10:24.6', '+16:08:38.3', '+14:39:38.3', '+16:14:57.9', '-09:03:26.7', '-09:13:46.2', '+00:42:15.4', '+15:01:05.0', '-00:07:18.6', '+00:44:35.0', '-09:46:45.1', '-10:29:23.0', '-08:52:17.7', '-09:13:36.5', '-00:46:03.3', '-09:58:02.9', '-10:59:50.7', '-09:12:52.4', '+16:12:15.0', '-09:01:19.1', '+14:42:01.2', '-08:59:22.3', '+16:10:54.4', '-09:01:29.6', '-00:08:47.0', '+00:10:56.0', '-00:55:31.5', '+14:51:49.8', '-00:52:32.0', '-10:39:57.3', '-10:27:02.8', '-00:43:07.3', '+14:14:37.3', '-08:59:56.6', '+15:39:57.8', '-00:43:29.5', '-10:30:17.6', '+14:14:12.7', '+00:02:12.3', '-00:17:13.7', '-09:22:09.9', '+13:55:45.7', '-09:31:58.0', '+15:39:18.3', '-10:46:44.8', '+00:02:17.3', '+15:35:32.2', '-09:48:14.8', '-00:29:30.8', '+00:32:46.7', '+00:59:54.6', '-10:32:28.2', '+15:39:14.6', '+14:44:35.7', '+00:17:23.7', '+00:21:03.0', '-01:01:09.6', '-00:12:45.4', '-00:33:56.1', '-00:14:06.6', '+00:29:31.6', '-01:14:05.7', '-09:32:42.0', '+14:17:25.9', '+15:04:09.3', '+14:45:45.3', '+16:05:02.0', '-10:53:17.5', '+00:14:35.6', '+14:29:11.2', '-00:50:50.1', '-00:19:20.4', '+14:18:49.5', '-10:39:58.2', '+15:50:57.3', '+14:59:58.6', '-00:24:46.7', '-10:45:59.4', '-10:37:58.6', '+15:33:56.2', '-09:03:42.4', '-10:57:50.7', '-10:56:00.0', '-10:41:43.4', '+14:31:48.8', '-09:20:11.6', '+00:05:48.2', '+15:39:39.2', '+14:13:58.2', '-09:11:24.5', '-10:23:56.0', '+16:04:50.6', '+14:44:03.0', '-00:45:09.6', '-00:46:22.5', '+00:11:01.3', '+14:41:57.7', '-00:07:22.7', '-10:40:39.7', '+00:49:08.5', '-08:50:05.1', '+00:40:04.5', '-01:03:53.1', '-09:29:11.7', '-09:09:23.4', '-10:44:28.6', '-09:17:03.2', '-09:25:47.5', '-09:18:11.3', '+00:28:35.0', '-10:49:57.9', '-09:28:15.8', '+14:36:20.2', '-08:45:50.8', '-09:54:42.2', '-09:30:53.8', '-09:17:31.2', '+00:58:37.2', '-00:17:57.1', '-09:13:49.5', '-09:13:46.4', '+14:08:02.7', '+15:16:02.2', '-10:39:56.2', '+15:12:52.5', '+15:11:10.2', '+15:47:53.1', '-00:59:13.9', '+01:01:31.8', '+00:00:08.6', '+00:07:40.0', '-00:49:16.7', '+15:30:43.4', '-10:10:09.0', '+01:00:29.2', '+15:31:05.7', '+00:40:22.5', '-00:51:26.8', '-09:45:14.0', '-09:52:26.1', '-10:17:43.3', '+00:22:07.0', '+00:14:45.9', '+00:46:45.0', '-08:47:32.9', '+00:59:05.5', '+15:33:58.9', '-09:00:58.9', '+00:34:11.1', '-01:06:11.4', '-11:03:22.2', '+15:59:33.7', '-08:58:58.4', '-09:00:23.7', '+14:33:48.4', '-09:29:28.6', '+14:34:30.6', '-01:03:27.8', '+00:41:21.3', '+00:15:48.5', '-00:48:00.4', '-00:40:38.4', '-10:14:19.2', '-00:44:46.1', '+00:07:55.2', '+15:53:10.4', '-09:17:54.0', '-09:26:52.2', '-10:59:33.5', '-00:22:23.9', '+00:57:04.0', '-10:57:02.4', '-09:56:26.5', '-10:17:33.2', '-09:59:08.4', '-00:52:44.9', '-00:57:54.9', '-01:07:00.9', '-00:57:11.8', '-01:14:30.6', '-09:37:56.3', '+13:52:34.8', '+00:02:37.8', '-00:47:32.7', '-01:09:33.1', '+15:17:38.9', '-00:34:56.3', '+00:25:49.5', '-09:31:30.9', '+14:53:14.6', '+14:53:20.7', '-00:45:06.5', '+00:31:28.3', '-08:46:30.7', '+00:25:30.9', '-10:56:11.9', '-10:54:20.0', '-08:52:52.4', '+00:16:44.9', '+13:40:09.5', '-00:25:57.8', '-09:47:43.7', '+14:50:12.4', '-10:02:45.2', '+00:43:13.0', '+15:59:22.6', '-01:06:39.4', '+15:53:12.1', '-01:10:36.4', '-08:57:47.0', '-00:48:47.0', '-00:52:23.2', '-00:57:33.1', '-00:11:14.7', '-00:19:12.4', '+15:36:35.0', '+15:41:53.2', '-09:42:32.5', '+13:33:47.0', '-09:25:04.4', '-08:56:43.2', '-09:48:46.1', '-08:36:12.7', '-10:38:32.0', '+00:31:22.8', '+00:54:35.4', '+13:27:40.9', '-09:27:12.0', '+00:39:40.8', '+15:10:39.5', '+00:38:43.2', '+00:08:49.7', '+14:12:21.8', '+15:36:42.2', '+14:29:30.2', '+01:02:53.0', '+00:19:34.8', '-00:24:59.1', '+15:05:13.6', '-09:16:44.2', '-00:02:45.7', '+01:08:51.8']
#SDSS superclean CS
sdss_list=[587730773888794890, 587727225690194001, 587727220876706008, 587727221413642408, 588015507658834108, 587727225690259639, 588015508732641374, 587727180060295268, 587727223561257129, 587727178986553395, 587730773888991473, 588015510343385169, 588015510343385206, 587727221413904560, 587727221413904728, 587730772815446262, 587730773889187885, 587727225153716451, 587730772815446274, 587731187278086304, 588015508732969059, 587727226227458180, 587727180597428330, 587731186741346415, 587727227838136451, 588015509806776498, 587730772815577232, 587727179523817590, 587727226764460180, 587727177913204856, 587730773889384611, 587727223024779503, 587727226227654787, 588015508196294817, 588015508733165680, 587727226227654761, 587731185667735754, 587727223024910524, 587731185130995898, 587730772815839344, 587730775500193994, 587731185667866816, 587727223024976038, 587730772815839449, 587727226764722299, 587727221951234166, 588015508196491400, 587727221414428872, 587731186741739641, 587730774426583149, 587731185667997882, 587727223025107185, 587727178987339822, 587727221951365269, 587730774963519495, 587730774963519732, 588015508196687961, 587727223025172713, 587727221951496296, 588015509807366248, 588015509270495431, 587730773889843359, 587731186205065323, 587731186205065344, 587731185131323634, 588015508733689984, 587727226765049986, 587727226765049913, 587727226228179083, 588015509807431861, 587730773353103492, 587727225154502764, 588015507660013666, 587727226765115492, 588015508196884558, 587727179524472926, 587727179524473006, 587731187815874690, 587727177913925789, 588015508733821094, 587731186742132985, 587731185131585717, 588290880564887657, 588015509270823089, 587727225154764940, 587730773890236642, 587727225691635884, 588290880565018738, 588015509270954201, 587731185668653188, 588015508197212201, 587731187279265981, 587731186742394983, 587731187279265994, 587727179524800681, 587731187279331410, 587731186205589586, 588015508734148754, 588290882175697097, 588015508734214258, 587727179524931685, 587730775501045957, 587730773353562270, 587731186205655240, 587731185668849799, 588015507660537942, 587730775501111498, 588015510344958052, 587730773353693301, 587731186205786214, 587731185668915245, 587731185132044450, 587727180061933748, 587730774427500730, 587727227839512712, 588015509271281831, 588290881102217439, 587727178451386525, 587727180062064786, 587731186742788238, 587730774964502681, 588015508197671078, 587727225155354763, 587727225155354652, 587727180062195851, 587730775501439168, 588290881639350458, 587727227302903934, 587731185669439607, 588015509271740514, 588290882176417925, 588015509271806052, 587730775501766791, 587731185132634206, 588290880565870747, 587727177915039886, 587730773354283222, 588290882176483496, 588290881102807165, 587730773354348704, 587727227840102493, 588015507661324401, 588290881102872703, 588290879492259979, 587730773354414192, 587727179525783739, 587731186206507199, 587730773891285162, 588290881639743547, 587730773354414249, 587731186743443616, 588015507661389943, 587727178452107412, 588290880566067265, 588290880566132939, 588015509808939099, 588015507661520918, 588290880566198454, 587730773354610868, 588015509809135731, 588015507661652183, 588015509809135916, 587727225156141259, 588290881640071382, 588015509809201238, 587727179526111377, 587727178989240486, 587731185670029524, 587730775502291109, 588290879492653215, 588015507661783172, 587727180063113363, 587727178452500610, 587731187280773305, 587727226766950508, 587724199885013029, 588015508735656067, 587724199885013129, 587731186743967943, 588015508735721557, 588015507661979748, 587731187280969914, 587731185670357175, 587727227840888987, 587731187817840787, 587731185133486290, 588015508198981794, 588015510346465416, 587731187817906345, 587731187817906359, 587727226767212721, 587731186744230004, 587727225156665442, 587731185670488278, 588015508199112844, 587727225156730966, 587727225693601912, 588015510346596507, 588015510346662008, 587731185133682765, 587727226767343762, 587731185133748384, 587724233173827648, 587731186207490173, 588015508736114824, 588015509272985710, 587727177916219465, 588015508199309451, 588015510346858598, 587727226767540279, 587731186207686803, 588015509273182266, 587727180600705119, 587727180063834204, 587731187281494216, 587731187281559664, 587724231563477157, 587731187818430564, 587727179527094384, 587727226230931521, 588015508736507988, 587727225157255237, 587727227304804420, 587731187818627249, 587724233711157418, 587731185134338213, 587731185671274603, 588015509273575602, 587727227304935542, 587727177916743773, 588015509810446476, 587724197201707115, 588015508199899235, 587727225694388294, 587731187818823860, 587727180601229418, 587724234248224914, 588015510347448339, 587727227305001011, 588015508736835649, 587724198275514505, 587731186745213014, 587724198275645522, 587727178990682279, 587731185671471206, 588015508736966677, 587727225157648439, 587727178990747862, 587724199886258329, 587731187282149429, 587724198812581994, 587731187282215080, 587731187819085863, 588015508737097743, 587724199349518484, 587727180064620686, 587731186745409694, 587727227842199670, 587731186745475231, 587724233711681696, 587727177917202487, 588015509274099894, 587724232101199896, 587727180064817249, 587731186745540822, 587724198275973252, 588015509274165442, 587724199886586000, 587727226231783500, 587727180601753667, 587731187819413572, 587724198276038851, 587724199349780643, 587727178991141022, 587731185135124558, 587727178454335591, 587731185671995534, 587724198276169809, 587727180065013845, 587727225158172847, 587727180065013869, 587724197202493564, 588015509811298500, 587724198813106377, 588015508737622060, 587724233712140421, 587724232638464033, 587731186745999456, 587724234249076842, 587724197739495456, 587724197202690182, 587727226232242263, 587724199887044764, 588015510348431508, 588015507664076945, 587727180065341564, 587724231564918985, 587727177917923465, 587727226232307811, 587727180065472682, 587727178991730776]
ra_list=['00:00:38.69', '00:00:44.79', '00:00:45.60', '00:00:52.91', '00:00:57.31', '00:01:06.37', '00:01:28.50', '00:01:51.29', '00:01:53.70', '00:02:06.34', '00:02:15.93', '00:02:21.24', '00:02:39.65', '00:03:25.64', '00:03:48.03', '00:03:58.80', '00:04:00.62', '00:04:02.52', '00:04:02.94', '00:04:03.27', '00:04:06.01', '00:04:10.46', '00:04:34.27', '00:04:52.36', '00:04:54.03', '00:05:05.45', '00:05:27.04', '00:05:30.43', '00:05:31.73', '00:05:42.53', '00:05:42.93', '00:05:50.86', '00:05:57.16', '00:06:03.91', '00:06:08.42', '00:06:14.51', '00:06:35.73', '00:07:22.15', '00:07:25.10', '00:07:38.00', '00:07:40.56', '00:07:42.14', '00:07:46.98', '00:07:55.05', '00:08:04.10', '00:08:05.63', '00:08:07.23', '00:08:26.16', '00:08:26.51', '00:08:41.55', '00:08:47.86', '00:09:03.09', '00:09:04.74', '00:09:08.63', '00:09:20.50', '00:09:44.98', '00:09:49.38', '00:09:57.79', '00:10:10.72', '00:10:12.70', '00:10:18.23', '00:10:22.17', '00:10:31.47', '00:10:39.34', '00:10:46.65', '00:10:51.93', '00:11:00.16', '00:11:03.29', '00:11:03.62', '00:11:08.98', '00:11:13.14', '00:11:14.38', '00:11:18.39', '00:11:19.84', '00:11:30.93', '00:11:46.45', '00:11:47.61', '00:12:03.93', '00:12:07.37', '00:12:08.63', '00:12:35.48', '00:13:08.82', '00:13:11.78', '00:13:30.49', '00:13:45.19', '00:13:47.14', '00:14:01.49', '00:14:25.73', '00:14:26.40', '00:14:26.49', '00:14:31.86', '00:14:52.45', '00:14:55.08', '00:14:57.60', '00:15:04.44', '00:15:05.15', '00:15:05.87', '00:15:09.48', '00:15:26.85', '00:15:39.87', '00:15:45.91', '00:15:55.40', '00:15:56.51', '00:16:10.24', '00:16:14.64', '00:16:15.28', '00:16:28.36', '00:16:43.64', '00:16:46.37', '00:16:54.92', '00:16:56.45', '00:17:08.77', '00:17:27.49', '00:17:34.86', '00:17:40.52', '00:17:41.74', '00:17:42.52', '00:17:49.09', '00:18:14.32', '00:18:21.01', '00:18:40.01', '00:18:56.80', '00:19:09.98', '00:19:20.84', '00:19:36.41', '00:19:44.69', '00:20:07.63', '00:20:09.29', '00:21:37.30', '00:21:41.02', '00:22:19.38', '00:22:19.86', '00:22:20.33', '00:22:20.80', '00:22:44.62', '00:22:50.29', '00:22:52.20', '00:22:57.10', '00:23:03.99', '00:23:09.31', '00:23:12.65', '00:23:33.88', '00:23:35.78', '00:23:39.55', '00:23:43.86', '00:23:47.65', '00:23:51.74', '00:23:58.83', '00:23:59.39', '00:24:05.42', '00:24:08.71', '00:24:16.24', '00:24:30.75', '00:24:42.70', '00:24:52.24', '00:24:55.45', '00:25:15.14', '00:25:38.16', '00:25:58.98', '00:26:14.77', '00:26:28.46', '00:26:38.45', '00:26:46.21', '00:27:02.22', '00:27:03.83', '00:27:08.28', '00:27:10.54', '00:27:18.29', '00:27:27.45', '00:27:42.25', '00:27:49.74', '00:28:17.87', '00:28:22.62', '00:28:33.80', '00:28:44.81', '00:28:44.83', '00:28:51.28', '00:28:59.79', '00:29:08.09', '00:29:21.55', '00:29:32.88', '00:30:12.18', '00:30:18.20', '00:30:29.83', '00:30:32.14', '00:30:33.49', '00:30:47.04', '00:30:53.61', '00:31:00.26', '00:31:04.12', '00:31:11.27', '00:31:22.50', '00:31:23.71', '00:31:41.57', '00:31:47.25', '00:31:56.77', '00:31:59.54', '00:32:05.26', '00:32:15.70', '00:32:20.84', '00:32:28.94', '00:32:39.05', '00:32:55.09', '00:32:55.67', '00:33:10.45', '00:33:14.82', '00:33:27.76', '00:33:51.61', '00:33:59.83', '00:34:02.79', '00:34:43.51', '00:34:46.22', '00:34:53.45', '00:35:02.91', '00:35:14.54', '00:35:45.38', '00:35:46.93', '00:35:47.00', '00:36:01.99', '00:36:28.92', '00:36:30.30', '00:36:52.41', '00:37:33.68', '00:37:45.90', '00:37:52.17', '00:38:15.56', '00:38:23.48', '00:38:28.13', '00:38:36.47', '00:38:38.49', '00:38:43.07', '00:38:44.94', '00:38:55.92', '00:39:20.13', '00:39:29.67', '00:39:30.96', '00:39:32.10', '00:39:34.83', '00:39:35.79', '00:39:37.84', '00:39:38.32', '00:40:13.35', '00:40:23.48', '00:40:29.06', '00:40:33.87', '00:40:37.85', '00:40:40.77', '00:40:44.65', '00:40:57.54', '00:41:10.87', '00:41:11.60', '00:41:36.30', '00:41:46.12', '00:41:49.30', '00:41:58.03', '00:42:05.03', '00:42:17.85', '00:42:21.90', '00:42:39.34', '00:42:47.55', '00:42:48.27', '00:43:25.92', '00:43:32.39', '00:43:38.76', '00:43:39.25', '00:43:45.19', '00:43:47.43', '00:44:04.98', '00:44:07.09', '00:44:11.73', '00:44:29.48', '00:44:42.40', '00:44:43.95', '00:44:50.64', '00:44:59.24', '00:45:00.49', '00:45:22.14', '00:45:22.30', '00:45:34.11', '00:45:34.68', '00:45:51.93', '00:45:56.27', '00:46:24.47', '00:46:31.55', '00:46:55.31', '00:46:58.97', '00:47:19.39', '00:47:25.00', '00:47:30.25', '00:47:35.13', '00:47:58.38', '00:48:18.88', '00:48:27.54', '00:48:31.67', '00:48:32.05', '00:48:54.46', '00:49:23.00', '00:49:25.29', '00:49:26.97', '00:49:56.03', '00:50:14.88']
dec_list=['+14:35:48.2', '-10:24:20.9', '+14:04:28.8', '+14:24:48.7', '-01:06:53.4', '-10:24:00.9', '-00:13:27.9', '-09:24:30.4', '+16:08:15.5', '-10:13:06.8', '+14:40:18.5', '+00:56:14.3', '+00:56:33.6', '+14:28:37.7', '+14:20:20.3', '+13:43:28.6', '+14:42:13.4', '-10:51:44.3', '+13:45:05.3', '+00:38:44.4', '-00:13:33.4', '-10:01:18.2', '-09:03:09.3', '+00:21:06.2', '-08:53:40.7', '+00:27:45.6', '+13:48:49.6', '-09:57:01.2', '-09:35:01.3', '-11:10:48.9', '+14:40:19.0', '+15:39:44.1', '-10:05:58.3', '-00:43:37.1', '-00:22:19.7', '-10:06:02.8', '-00:34:15.9', '+15:38:11.5', '-00:51:46.2', '+13:55:46.8', '+15:51:02.4', '-00:34:15.0', '+15:40:52.1', '+13:48:45.9', '-09:36:42.4', '+14:50:23.3', '-00:46:47.2', '+14:28:33.7', '+00:22:09.4', '+15:10:59.4', '-00:37:30.4', '+15:38:42.7', '-10:19:25.9', '+14:56:59.2', '+15:32:34.4', '+15:27:12.4', '-00:41:03.2', '+15:38:45.0', '+14:47:04.2', '+00:29:38.4', '+00:12:23.7', '+14:37:57.5', '-00:02:34.5', '-00:03:10.4', '-00:53:32.5', '-00:22:06.3', '-09:37:28.2', '-09:39:04.2', '-10:02:14.6', '+00:34:26.5', '+14:16:53.8', '-11:01:59.0', '-01:14:34.5', '-09:39:40.6', '-00:43:53.7', '-09:59:01.4', '-09:55:04.8', '+01:06:19.2', '-11:12:04.9', '-00:14:39.9', '+00:13:54.1', '-01:00:01.2', '+14:53:40.9', '+00:07:37.3', '-10:56:11.6', '+14:46:26.1', '-10:35:58.4', '+14:56:14.5', '+00:06:51.8', '-00:25:58.1', '-00:44:15.2', '+00:38:03.3', '+00:15:08.2', '+00:39:07.0', '-09:50:42.4', '+00:38:34.6', '-00:00:22.6', '-00:21:59.5', '+16:14:07.0', '-00:16:01.5', '-09:51:46.2', '+16:00:13.3', '+14:11:51.0', '-00:04:05.6', '-00:26:06.9', '-01:12:13.9', '+16:01:31.2', '+00:52:50.5', '+14:19:15.1', '-00:05:17.6', '-00:26:12.4', '-00:57:28.9', '-09:34:27.2', '+15:05:25.0', '-08:54:58.2', '+00:07:52.3', '+15:20:09.7', '-10:47:03.7', '-09:23:14.8', '+00:24:17.5', '+15:33:52.9', '-00:40:42.0', '-10:53:08.7', '-10:56:38.8', '-09:26:43.3', '+15:51:30.5', '+15:47:44.1', '-09:14:40.5', '-00:28:28.2', '+00:01:35.3', '+16:06:20.9', '+00:00:32.8', '+15:56:30.4', '-00:56:03.9', '+14:56:59.1', '-11:11:33.5', '+14:11:29.1', '+16:15:11.2', '+15:22:34.6', '+14:14:55.5', '-08:56:44.2', '-01:12:44.4', '+15:14:07.1', '+14:06:34.6', '+14:18:24.2', '-09:47:26.4', '-00:07:42.7', '+14:40:12.8', '+15:46:13.5', '+14:10:23.7', '+00:22:44.7', '-01:06:36.2', '-10:42:20.7', '+14:49:28.5', '+14:48:04.2', '+00:31:48.7', '-01:06:42.3', '+14:49:48.5', '+14:20:52.6', '+00:32:46.8', '-01:07:51.1', '+00:33:33.7', '-10:57:38.8', '+15:45:57.9', '+00:27:37.3', '-09:56:08.7', '-10:19:20.4', '-00:33:54.6', '+15:51:02.2', '+14:03:39.8', '-01:12:00.0', '-09:29:34.3', '-10:40:36.8', '+00:48:50.6', '-09:44:17.4', '+16:00:58.9', '-00:17:09.6', '+15:58:14.1', '+00:16:46.2', '-00:22:50.4', '-01:05:44.5', '+00:41:13.9', '-00:30:08.1', '-08:46:59.9', '+01:10:46.9', '-00:54:43.1', '-00:49:13.6', '+01:01:32.6', '+01:14:38.3', '+01:05:49.0', '-09:42:07.0', '+00:18:43.8', '-11:01:31.2', '-00:32:14.4', '-00:43:59.4', '-11:00:56.9', '-10:36:46.1', '+00:59:37.7', '+00:54:22.2', '-00:57:45.1', '-09:34:38.0', '-00:55:52.5', '+15:15:22.4', '-00:01:15.7', '-00:21:00.8', '+00:12:03.7', '-11:06:12.4', '-00:44:32.2', '+00:57:09.3', '-09:42:19.2', '-00:02:26.7', '+00:06:51.7', '-08:55:57.8', '-09:22:06.3', '+00:41:45.5', '+00:41:50.2', '+13:59:01.5', '+01:08:05.1', '-09:53:18.7', '-10:06:22.2', '-00:17:14.6', '-10:52:09.9', '-09:13:08.3', '+01:05:38.8', '+15:36:15.3', '-00:58:02.6', '-00:36:28.7', '+00:07:20.8', '-09:11:45.0', '-11:02:49.3', '+00:34:51.8', '+13:53:41.9', '-00:48:21.8', '-10:28:55.5', '+01:03:50.0', '-09:02:22.8', '+16:05:01.4', '+00:51:35.9', '-09:11:40.3', '-00:19:43.3', '+14:39:51.2', '+00:20:42.0', '+14:36:49.4', '-10:18:19.7', '-00:34:23.8', '-00:19:50.4', '-10:55:29.8', '-10:17:12.1', '+15:49:18.6', '+00:39:18.1', '+15:03:09.7', '+00:46:38.2', '+01:10:37.4', '-00:24:37.4', '+15:21:21.2', '-09:32:04.0', '+00:22:14.1', '-08:44:37.4', '+00:16:38.7', '+15:42:51.1', '-11:10:38.9', '+00:11:54.1', '+14:20:33.2', '-09:31:20.8', '+00:12:51.2', '+14:32:43.7', '+00:03:34.6', '+15:48:42.7', '-10:03:45.4', '-08:58:56.8', '+01:06:30.3', '+14:42:10.9', '+15:24:01.9', '-10:21:22.6', '-01:01:24.3', '-10:34:36.3', '-00:26:06.3', '+14:34:00.0', '-09:22:58.2', '-10:59:14.0', '-09:19:41.6', '+13:48:43.7', '+00:32:50.5', '+14:59:06.0', '-00:22:17.4', '+15:35:33.7', '+14:42:12.6', '+00:20:13.8', '+16:08:13.0', '+14:09:18.4', '+13:46:49.2', '-09:58:01.5', '+15:46:46.8', '+01:00:48.1', '-01:06:12.2', '-09:25:09.4', '+13:56:00.9', '-11:09:50.5', '-10:06:22.6', '-09:26:01.8', '-10:11:35.9']
#ra=155.85656923, dec=32.73012414,   ObjId = 587738615409737779
#102325.57+324348.4
#115822.57+323102.1
#GALAXY    ra=179.59405488, dec=32.51726569,   ObjId = 587739609162317959
#SDSS J162345.20+080851.1
#GALAXY    ra=245.93834084, dec=8.1475281,   ObjId = 587742610277532137
#sdss_list=[587738615409737779,587739609162317959,587742610277532137]
#ra_list=['10:23:25.57','11:58:22.57','16:23:45.20']
#dec_list=['+32:43:48.4','+32:31:02.1','+08:08:51.1']

merger=np.zeros(len(sdss_list))
kept_ids=[]#if you need a list of which ones ran without failing

print(len(sdss_list))


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

file=open('LDA_img_statmorph_SDSS_superclean_CS.txt','w')



counter=0

for i in range(len(sdss_list)):
    
    sdss=sdss_list[i]
    print('~~~~~~~~~~~~~~~~~~')
    print(sdss)
    print('~~~~~~~~~~~~~~~~~~')
    '''Search through drpall for redshift, the mangaid, and the designid, you need these to directly
    access the preimaging files, which are not stored in the same format as our input lists'''
    
    
    decode=SDSS_objid_to_values(sdss)
    #return skyVersion, rerun, run, camcol, field, object_num
    #https://data.sdss.org/sas/dr14/eboss/photoObj/frames/301/2566/2/

    
        
    try:
        if decode[4] > 100:
            im=fits.open('sdss/frame-r-00'+str(decode[2])+'-'+str(decode[3])+'-0'+str(decode[4])+'.fits')
        if decode[4] < 100:
            im=fits.open('sdss/frame-r-00'+str(decode[2])+'-'+str(decode[3])+'-00'+str(decode[4])+'.fits')
    except FileNotFoundError:
        os.chdir(os.path.expanduser('/Users/beckynevin/Documents/Backup_My_Book/My_Passport_Backup/MergerMonger/sdss'))
        '''Here is where it is necessary to know the SDSS data password and username'''
        try:
            
            if decode[4] > 100:
                os.system('wget https://data.sdss.org/sas/dr14/eboss/photoObj/frames/301/'+str(decode[2])+'/'+str(decode[3])+'/frame-r-00'+str(decode[2])+'-'+str(decode[3])+'-0'+str(decode[4])+'.fits.bz2')
                os.system('gunzip frame-r-00'+str(decode[2])+'-'+str(decode[3])+'-0'+str(decode[4])+'.fits.bz2')
            if decode[4] < 100:
                os.system('wget https://data.sdss.org/sas/dr14/eboss/photoObj/frames/301/'+str(decode[2])+'/'+str(decode[3])+'/frame-r-00'+str(decode[2])+'-'+str(decode[3])+'-00'+str(decode[4])+'.fits.bz2')
                os.system('gunzip frame-r-00'+str(decode[2])+'-'+str(decode[3])+'-00'+str(decode[4])+'.fits.bz2')
        except FileNotFoundError:
            continue
        os.chdir(os.path.expanduser('/Users/beckynevin/Documents/Backup_My_Book/My_Passport_Backup/MergerMonger'))


    try:
        if decode[4] > 100:
            im=fits.open('sdss/frame-r-00'+str(decode[2])+'-'+str(decode[3])+'-0'+str(decode[4])+'.fits')
        if decode[4] < 100:
            im=fits.open('sdss/frame-r-00'+str(decode[2])+'-'+str(decode[3])+'-00'+str(decode[4])+'.fits')
    except FileNotFoundError:
        STOP
    
    #Now you need to cut out around the object
    from astropy.coordinates import SkyCoord
    from astropy.wcs import WCS
    from astroquery.sdss import SDSS
    from astropy import coordinates as coords
    from astropy import units as u
    
    
    obj_coords = SkyCoord(str(ra_list[i])+' '+str(dec_list[i]),unit=(u.hourangle, u.deg))

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
    camera_data=(np.fliplr(np.rot90(stamp_a.data))/0.005)
    # Plot the the stamp in band Z with contours from band A
    #ax = plt.subplot(projection = wcs_a)
    #plt.imshow(camera_data*0.005, norm=matplotlib.colors.LogNorm(vmin=10**-1.5), cmap='afmhot')#, cmap='gray', origin='lower', alpha=1)
    #plt.colorbar()
    #plt.axis('off')
    #plt.savefig('sdss/'+str(sdss)+'.pdf')
    #continue
    
    
  #this is the r-band image, and we divide by the conversion factor for MaNGA, 0.005
    #We want things to be in units of counts
    #This is the average conversion between nmgy and counts for SDSS, but Source Extractor and Galfit only
        #require a rough amount
    
    
    #STOP
    
    #camera_data_ivar=abs(im[5].data)
    #calculate the percent error to then convert to counts:
    
    #percent_e=abs((1/camera_data_ivar**2)/im[4].data)
    camera_data_sigma = np.sqrt(camera_data)
    
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

    continue
    

     
    
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
    redshift=0.4
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
        
    plt.clf()
    plt.imshow(segm)#, origin='lower', cmap='gray')
    plt.colorbar()
    plt.savefig('../MaNGA_Papers/Paper_I/segmap_0_'+str(sdss)+'.pdf')

    label = np.argmax(segm.areas) + 1
    segmap = segm.data == label

    plt.clf()
    plt.imshow(segmap)#, origin='lower', cmap='gray')
    #plt.colorbar()
    plt.savefig('../MaNGA_Papers/Paper_I/segmap_1_'+str(sdss)+'.pdf')
 
    segmap_float = ndimage.uniform_filter(np.float64(segmap), size=10)

    plt.clf()
    plt.imshow(segmap_float)#, origin='lower', cmap='gray')
    plt.colorbar()
    plt.savefig('../MaNGA_Papers/Paper_I/segmap_2_'+str(sdss)+'.pdf')
    #segmap = segmap_float >  0.5
    #if limiting_sb < bg_level:
    print('limiting sb', limiting_sb)
    print('bg_level', bg_level)
        #segmap = segmap_float > (0.5*(np.max(segmap_float)-np.min(segmap_float))+np.min(segmap_float))
        #detect sources again
    new_threshold = (0.75*(np.max(segmap_float)-np.min(segmap_float))+np.min(segmap_float))
    segm = photutils.detect_sources(segmap_float, new_threshold, npixels, connectivity=8)

    plt.clf()
    plt.imshow(segm)
    plt.colorbar()
    plt.savefig('../MaNGA_Papers/Paper_I/segmap_3_'+str(sdss)+'.pdf')
            
    label = np.argmax(segm.areas[1:]) + 1
    segmap = segm.data == label

    plt.clf()
    plt.imshow(segmap)
    plt.savefig('../MaNGA_Papers/Paper_I/segmap_4_'+str(sdss)+'.pdf')
            
    segmap_float = ndimage.uniform_filter(np.float64(segmap), size=10)
    segmap = segmap_float >  0.5
    #else:
        #segmap = segmap_float >  0.5


    plt.clf()
    plt.imshow(np.flipud(segmap), origin='lower', cmap='gray')
    plt.savefig('../MaNGA_Papers/Paper_I/segmap_OG_'+str(sdss)+'.pdf')

    plt.clf()
    plt.imshow(camera_data, norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.savefig('../MaNGA_Papers/Paper_I/segmap_camera_data_'+str(sdss)+'.pdf')

    if i > 2:
        STOP
    else:
        continue
    

    threshold = photutils.detect_threshold(camera_data, snr=2, background=0.0)#was 1.5
    npixels = 5  # minimum number of connected pixels was 5
    segm = photutils.detect_sources(camera_data, threshold, npixels)

    label = np.argmax(segm.areas[1:]) + 1
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
                                                                                                                                                      

    #segm = photutils.detect_sources(masked, threshold, npixels)
    

    #seg=np.ones((np.shape(camera_data)[0],np.shape(camera_data)[1]))
    
    #label = np.argmax(segm.areas[1:]) + 1
    #segmap = segm.data == label
    plt.clf()
    plt.imshow(segmap, origin='lower', cmap='gray')
    plt.savefig('../MaNGA_Papers/Paper_I/segmap_OG_'+str(sdss)+'.pdf')

    STOP

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
    
    
    

    #call statmorph
    source_morphs = statmorph.source_morphology(camera_data, segmap, gain=100, skybox_size=10)#,weightmap=camera_data_sigma)#psf=psf,
    #try:
#        morph=source_morphs[0]
#    except IndexError:
#        continue
    
    
    morph=source_morphs[0]
    if morph.flag==1:
        '''This means the measurement is unreliable'''
        continue
    print('ellipticity =', morph.ellipticity)
    print('elongation =', morph.elongation)
    print('orientation =', morph.orientation)
    print('rpetro_circ =', morph.rpetro_circ)
    print('rpetro_ellip =', morph.rpetro_ellip)
    print('rhalf_circ =', morph.rhalf_circ)
    print('rhalf_ellip =', morph.rhalf_ellip)
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

    




    
    if counter==0:
        file.write('Counter'+'\t'+
            'ID'+'\t'+'Merger?'+'\t'+'# Bulges'+'\t'+'Sep'+'\t'+'Flux Ratio'+'\t'+'Gini'+'\t'+'M20'+'\t'+'C'+'\t'+'A'+'\t'+'S'+
                    '\t'+'Sersic n'+'\t'+'A_s'+'\t'
                   +'Elongation'+'\t'+'Sersic AR'+'\n')#was str(np.shape(vel_dist)[1]-1-j)                                                                             

    #The Merger? column is meaningless; MergerMonger will classify while ignoring these columns
    #but it is important to leave it in for ease so that the dataframes have the same headers and match formats
    file.write(str(counter)+'\t'+str(sdss)+'\t'+str(merger[i])+'\t'+str(num_bulges)+'\t'+str(sep)+'\t'+str(flux_r)+
                '\t'+str(gini)+'\t'+str(m20)+'\t'+str(con)+'\t'+str(asy)+'\t'+str(clu)+'\t'+str(ser)+'\t'+str(n)+'\t'+
               str(morph.elongation)+'\t'+str(inc)+'\n')#was str(np.shape(vel_dist)[1]-1-j)                                                                             
    if counter == 49:
        break
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
print('Finished and Congrats!')
    
