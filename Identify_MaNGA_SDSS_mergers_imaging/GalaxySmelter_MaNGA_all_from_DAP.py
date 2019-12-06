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
    print(im[0].header)
    STOP
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

    

    
    
    camera_data_ivar=abs(im[5].data)
    #calculate the percent error to then convert to counts:
    STOP
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



    
