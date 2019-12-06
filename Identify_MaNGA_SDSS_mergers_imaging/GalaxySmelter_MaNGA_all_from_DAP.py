##  GalaxySmelter - This is the 'Real' edition to be applied to a real galaxy survey to extract predictors and place them in a table.

# Things to install: Source Extractor, Galfit, statmorph (cite Vicente Rodriguez-Gomez)

# The first step is to determine how to obtain galaxy images, ivar images, and psf; here I use the table of MaNGA ids,
# table_manga_gals.txt, which contains the relevant information to go and wget all of these images
# utilizing wget - you can make your own directory that contains image files


# The code puts these images in units of counts and then uses all the tools from Metals.py to make a table
# of the value of all of the imaging predictors for each galaxy. They can then be classified as merging or not using
# MergerMonger...py

# Here, I run everything from within a folder that has subfolders imaging/ and preim/
# which contain the output check images and the input images, respectively.


# First, import things, these first few lines are so that this version of python plays nicely with plotting on the supercomputer

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
# And all the ingredients to measure imaging predictors from Metals.py:
from Metals import write_sex_default, run_sex, sex_params_galfit, plot_sex, write_galfit_feedme, run_galfit, galfit_params, input_petrosian_radius, make_figure, img_assy_shape
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astroquery.sdss import SDSS
from astropy import coordinates as coords
from astropy import units as u
from astropy.nddata import Cutout2D
import scipy.ndimage as ndi
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground

# This code snippet is really useful to get all the numbers in order to wget SDSS images from just the objid:
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



 
# Let the Smelting begin

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


# Alternately, you can supply this code directly with objids, ra_list, and dec_list:
#SDSS superclean merger Galaxy Zoo
#sdss_list=[587730773351923978, 587727177912680538, 587727223561257225, 587731187277955213, 587730774962864138, 587727178986750088, 588015510343712908, 587730775499997384, 587727180060688562, 587727225690783827, 587727226227720294, 587731186204868616, 588015510344040607, 588015508733624406, 587727225691242683, 587730772816167051, 588015510344302646, 588015509807497271, 587727178450927744, 588290881638760664, 587727178988191864]
#ra_list=['00:00:20.24', '00:00:37.17', '00:02:27.29', '00:02:49.07', '00:03:08.23', '00:03:57.92', '00:05:27.39', '00:05:54.09', '00:05:58.00', '00:06:07.51', '00:06:24.76', '00:08:27.81', '00:08:32.88', '00:10:11.15', '00:10:33.29', '00:10:55.50', '00:11:08.83', '00:11:43.68', '00:13:30.75', '00:14:31.15', '00:16:54.99']
#dec_list=['+14:11:09.9', '-11:02:07.6', '+16:04:42.5', '+00:45:04.8', '+15:33:48.8', '-10:20:44.3', '+00:50:48.1', '+15:53:42.8', '-09:21:17.8', '-10:30:32.7', '-10:01:31.2', '-00:00:17.7', '+01:02:20.1', '-00:14:30.8', '-10:36:10.9', '+13:52:49.7', '+00:50:43.7', '+00:31:22.5', '-10:43:17.6', '+15:49:02.2', '-10:23:44.0']


merger=np.zeros(len(df))
kept_ids=[]#if you need a list of which ones ran without failing

print(len(df))





file=open('LDA_img_statmorph_MaNGA_mytable.txt','w')



counter=0

# I would like to look at the distributions of AB mags and S_N after all is said and done:
mag_list=[]
S_N_list=[]

for i in range(len(df)):
    sdss=df['PLATEIFU'][i]
  
    
    # I have already placed all the run, camcol, etc values in the .txt file, but if you need to look them
    # up use the following line:
    #decode=SDSS_objid_to_values(sdss)
    #https://data.sdss.org/sas/dr14/eboss/photoObj/frames/301/2566/2/

    print('run', df['run'][i], 'camcol', df['camcol'][i], 'field', df['field'][i], 'redshift', df['redshift'][i])
    
    # First see if the images already exist in your file structure (they do in mine, I don't want to download them
    # each and every time):
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
    # If not there, then go and switch to your 'sdss' directory and go and wget the files from eboss:
    except FileNotFoundError:
        
        os.chdir(os.path.expanduser('/Users/beckynevin/CfA_Code/Kinematics_and_Imaging_Merger_Identification/Identify_MaNGA_SDSS_mergers_imaging/sdss'))
        # I'm not sure if it will prompt for the SDSS username here
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

    # Now load up the appropriate image:
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
    # This is the datamodel for SDSS frame images: https://data.sdss.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
    
    # We would like to have an error image and an image using three extensions of im:
    ''';; 0. find filename of the frame file
    framename = (sdss_name('frame', run, camcol, field, $
                           filter=filternum(filter), rerun=rerun))[0]+'.bz2'

    ;; 1. read in the FITS image from HDU0; the resulting image will be
    ;;    sky-subtracted as well as calibrated in nanomaggies/pixel
    img= mrdfits(framename,0,hdr)
    nrowc= (size(img,/dim))[1]

    ;; 2. read in sky, and interpolate to full image size; this returns a
    ;;    sky image the same size as the frame image, in units of counts
    sky= mrdfits(framename,2)
    simg= interpolate(sky.allsky, sky.xinterp, sky.yinterp, /grid)

    ;; 3. read in calibration, and expand to full image size; this returns
    ;;    a calibration image the same size as the frame image, in units of
    ;;    nanomaggies per count
    calib= mrdfits(framename,1)
    cimg= calib#replicate(1.,nrowc)
    Steps (0) and (1) just read in the "image". Step (2) reads in the sky HDU, and bilinearly interpolates "allsky" onto a 2048x1489 sized array at the points on the grid defined by "xinterp" and "yinterp". Step (3) reads in the 2048-element vector defined the calibration-times-flat-field for each row, and expands it to a full-sized image.

    If you have performed the above calculations, you can return the image to very close to the state it was in when input into the photometric pipeline, as follows:

    dn= img/cimg+simg
    These dn values are in the same units as the "data numbers" stored by the raw data files that come off the instrument. They are related to the detected number nelec of photo-electrons by:

    nelec= dn*gain
    The number of photo-electrons is the quantity that is statistically Poisson distributed. In addition, there are additional sources of noise from the read-noise and the noise in the dark current, which we will lump together here as the "dark variance." Thus, to calculate per-pixel uncertainties, you need the gain and darkVariance for the field in question. The darkVariance comes from the read noise and the noise in the dark current. In fact, these are nearly fixed as a function of camcol and filter (see the table below). You can retrieve the values from the field table in CAS (or the photoField file). With those values in hand the following yields the errors in DN:

    dn_err= sqrt(dn/gain+darkVariance)
    
    
    sky = im[2].data
    simg = np.interpolate()
    
    calib = im[1].data
    cimb = calib'''
    # For right now, I'm assuming there is no background sky and just taking the error to be the square root
    # of the image (which is like in the high <S/N> regime). I need to go back and fix this
    
    # Now you need to cut out around the object in a square 80 by 80 arcsec box (same as used for simulated galaxies):

    obj_coords = SkyCoord(str(df['RA'][i])+' '+str(df['DEC'][i]),unit=(u.hourangle, u.deg))

    size = u.Quantity((80,80), u.arcsec)
    wcs_a = WCS(im[0].header)

    stamp_a = Cutout2D(im[0].data, obj_coords, size, wcs=wcs_a)#was image_a[0].data
    
    
    
    # Use the header to convert from weird units of nanomaggies into Counts
    camera_data=(np.fliplr(np.rot90(stamp_a.data))/im[0].header['NMGY'])
    # Optionally Plot the the stamp to double check that it is working:
    

    plt.clf()
    plt.imshow(camera_data, norm=matplotlib.colors.LogNorm(vmin=10**0, vmax=10**4))
    #plt.title(np.min(AB_mag))
    plt.colorbar()
    plt.savefig('imaging/SDSS_counts_'+str(sdss)+'.png')


    # You can also convert to magnitudes from stamp_a's units, which are nanomaggies
    # units are nmgy, BUNIT, 1 nmgy = 3.631e-6 Jy
    # SDSS ugriz magnitudes are on the AB system where a magnitude 0 object has the same counts as a source with
    # F_nu = 3631 Jy
    Jy = (np.fliplr(np.rot90(abs(stamp_a.data))))*3.631e-6
    AB_mag = -2.5*np.log10(Jy/3631)

    plt.clf()
    plt.imshow(AB_mag)
    plt.title(np.min(AB_mag))
    plt.colorbar()
    plt.savefig('imaging/SDSS_AB_mag_'+str(sdss)+'.png')

    

    
    
    
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

    
    

     
    
    # I need to check this - I'm not convinced
    # Galfit needs a PSF file, and I'm not sure how to find the PSF file
    # although I could easily make one myself or feed it 1.5":
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


    # 24 μm; 0.396 arcseconds pixel-1
    pixelscale =  0.396
    redshift = df['redshift'][i]
    kpc_arcmin=cosmo.kpc_proper_per_arcmin(redshift)#insert the redshift                                                                                            
    # Divide the pixelscale (arcsec) by kpc/arcsec to get kpc size of pixels
    kpc_pix=pixelscale/(kpc_arcmin.value/60)#This is the number of kpc/pixel
    
    # Running sextractor before galfit to determine good guess input parameters
    

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
    
    
    

    # Now put together all of the necessary Galfit inputs - it is a little finnicky and needs to know
    # magnitude guesses, which is annoying because everything is in counts and often there's no exposure time
    # provided, but you can guess the magnitude as below by setting the mag_zpt and then calculating the flux ratio
    # given the source extractor background and main source counts:

    
 

    # This is scaled up using the background flux and the flux from the petrosian output
    
    # Now use the output of source extractor to also determine the petrosian radius,
    # the semimajor and semiminor axes, the flux guess, the bg level, the PA, and the axis ratio
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
        

    # This code runs galfit inline and creates an output file
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
    
    # This extracts useful parameters from Galfit such as the kpc separation if there are two nucli, the flux ratio (if there
    # are two nuclei), the sersic index, and the inclination of the main galaxy:
    h=galfit_params(sdss,num_bulges,kpc_pix)
    if h[2]==999:
        continue


    sep=h[0]
    flux_r=h[1]
    ser=h[2]# you can use many of these predictors to check the output, but the only important one is the sersic profile
    inc=h[3]


    # I think the most complicated part of this analysis is creating a segmentation map.
    # The segmentation map is used to calculate many of the imaging predictors and it is important
    # because it helps decide which pixels belong to the main galaxy.
    # As a side note, one relative advantage on this approach over CNN approaches to identify mergers
    # is that this approach does not consider ALL pixels like a CNN. Instead it uses the seg map to
    # determine which pixels it cares about. This means it is less sensitive to having observational
    # realism in terms of background galaxies than many CNNs.
    
    # Find the surface brightness at the petrosian radius, which is used to threshold the image:

    #limiting_sb=input_petrosian_radius(r_sex, b_sex, flux_sex)
    #print('limiting sb at petrosian', limiting_sb)

    #threshold = np.array(limiting_sb*np.ones((np.shape(camera_data)[0],np.shape(camera_data)[1])))
  
    #npixels = 5  # 1 is minimum number of connected pixels

    
    #segm = photutils.detect_sources(camera_data, threshold, npixels)
        
    # You can either make the seg map this way or you can use snr below, which
    # I find works better for SDSS:

    threshold = photutils.detect_threshold(camera_data, snr=1.5)#, background=0.0)#was 1.5
    npixels = 5  # minimum number of connected pixels was 5
    segm = photutils.detect_sources(camera_data, threshold, npixels)

    
    label = np.argmax(segm.areas) + 1
    segmap = segm.data == label
    

    segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
    segmap = segmap_float > 0.5

    
    
    
    # In case you need to do background estimation:
    
    sigma_clip = SigmaClip(sigma=3., iters=10)
    bkg_estimator = MedianBackground()
    try:
        bkg = Background2D(camera_data, (np.shape(camera_data)[0],np.shape(camera_data)[1]),edge_method='pad',
                           filter_size=(3,3),sigma_clip=sigma_clip,bkg_estimator=bkg_estimator)
    except ValueError:
        continue

    
    
    

    # call statmorph (cite Vicente Rodriguez-Gomez)
    source_morphs = statmorph.source_morphology(camera_data, segmap, 
                                                weightmap=camera_data_sigma,skybox_size=20)#,weightmap=camera_data_sigma)#psf=psf,
    try:
        morph=source_morphs[0]
    except IndexError:
        print('STATMORPH FAIL')
        continue
    
    
    morph=source_morphs[0]
    
    if morph.sn_per_pixel < 2.5:
        # This is how to weed out galaxies with too low of S/N per pixel:
        # Vicente says that below a certain level the predictors got wonky and I found
        # this to be true at <S/N> < 2.5
        continue
    if morph.flag==1:
        # This means the measurement is unreliable
        continue
    print('S_N', morph.sn_per_pixel, 'mag', mag_flux)

    # This is a nifty diagnostic plot that shows statmorph at work for each galaxy:
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
    
    # Most of the predictors are from statmorph
    gini=morph.gini
    m20=morph.m20
    con=morph.concentration
    asy=morph.asymmetry
    clu=morph.smoothness
    
    # I have my own code to measure shape asymmetry, but I wound up using the value from
    # statmorph and found they were similar in relative value between different galaxies:
    #n = img_assy_shape(segmap,sdss)
 

    
    n = morph.shape_asymmetry
    if ser > 20:
        ser =morph.sersic_n
    
    if counter==0:
        file.write('Counter'+'\t'+
            'ID'+'\t'+'Merger?'+'\t'+'# Bulges'+'\t'+'Sep'+'\t'+'Flux Ratio'+'\t'+'Gini'+'\t'+'M20'+'\t'+'C'+'\t'+'A'+'\t'+'S'+
                    '\t'+'Sersic n'+'\t'+'A_s'+'\n')#was str(np.shape(vel_dist)[1]-1-j)

    # The Merger? column is meaningless; MergerMonger will classify while ignoring these columns
    # but it is important to leave it in for ease so that the dataframes have the same headers and match format
    # with the simulated galaxies, where this label is actually important.
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



    
