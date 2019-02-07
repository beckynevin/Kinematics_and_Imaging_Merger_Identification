
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
import os
import numpy as np
from scipy import ndimage
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage import iterate_structure
from scipy.ndimage.filters import maximum_filter
import sys
import matplotlib
from numpy import shape
import scipy
import scipy.optimize as opt
from astropy.cosmology import WMAP9 as cosmo
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from astropy.io import fits
import math
import photutils
import statmorph
from skimage import measure

'''imports a million things'''
import pyfits
from photutils import CircularAperture,aperture_photometry

'''This code block reads in the broadband images and then will open the broadband extension,
...
'''
def produce_camera(img, viewpoint):
    im=pyfits.open(img)

    camera_data=im['CAMERA'+str(viewpoint)+'-BROADBAND'].data
    pixelscale =  im['CAMERA'+str(viewpoint)+'-BROADBAND'].header['CD1_1']

    return pixelscale, camera_data

def determine_coords(img):
    
    '''Apply a 10x10 kernal to the image to filter out noise (its basically a low pass filter)
    to smooth things out'''
    kernel = np.ones((10,10))

    lp = ndimage.convolve(img, kernel)#was result
    
    
    '''Okay here is where you can filter out the really low stuff
    (anything below 20% of the max is eliminated so that we can detect the peak pixel)'''
    
    max_value=(lp.max())
    low = np.where(lp < 0.2*max_value)
    
   
    lp[low] = 0
    
    
    
    
    
    
    '''Detects brightest peaks in an image (can detect more than 1)'''
    indices = np.where(detect_peaks(lp) == 1)#was hp_lp_sharp
    
    number_of_sols=len(indices[0])
    
    
    try:
        return indices[0][0],indices[0][-1],indices[1][0],indices[1][-1], lp, number_of_sols
    except IndexError:
        #if there are no peaks this means the simulation was somehow cut off and
        #starting with returning zeros will flag the entire procedure to continue
        #without further ado
        return 0,0,0,0,lp,number_of_sols

def detect_peaks(image):
    """
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    struct = generate_binary_structure(2,1)
    
    neighborhood = iterate_structure(struct, 10).astype(bool)
    
    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.
    
    
    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

'''
Fits 2D gaussians to surface brightness using the guesses from the low pass filter of the galaxy locations
Basically, this is my own design of a rough precursor to Source Extractor
'''
def fit_2_gaussian(x_1,y_1,x_2,y_2, data):
    # Create x and y indices
    data=np.flipud(data)
    x = np.linspace(0, 299, 300)
    y = np.linspace(0, 299, 300)
    x, y = np.meshgrid(x, y)
    

    # add some noise to the data and try to fit the data generated beforehand
    initial_guess = (20,x_1,y_1,7,7,0,10,20,x_2,y_2,7,7,0)#these are good guesses for the units of surface brightness
    data=data.ravel()
    
   
    
    try:
        popt, pcov = opt.curve_fit(twoD_two_Gaussian, (x, y), data, p0=initial_guess)
        fit='yes'
    except RuntimeError:
        popt=[0,0,0,0,0,0,0,0,0,0,0,0,0]
        fit='no'
        #flag for if the fit failed
 
    
    return popt[1], popt[2], popt[8], popt[9], popt[0], popt[7], np.sqrt(popt[3]**2+popt[4]**2), np.sqrt(popt[10]**2+popt[11]**2), fit 


def twoD_two_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset,
                     amplitude_2, xo_2, yo_2, sigma_x_2, sigma_y_2, theta_2):
    (x, y) = xdata_tuple 
    xo = float(xo)
    yo = float(yo)   
    xo_2 = float(xo_2)
    yo_2 = float(yo_2)  
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    
    a_2 = (np.cos(theta_2)**2)/(2*sigma_x_2**2) + (np.sin(theta_2)**2)/(2*sigma_y_2**2)
    b_2 = -(np.sin(2*theta_2))/(4*sigma_x_2**2) + (np.sin(2*theta_2))/(4*sigma_y_2**2)
    c_2 = (np.sin(theta_2)**2)/(2*sigma_x_2**2) + (np.cos(theta_2)**2)/(2*sigma_y_2**2)
    
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))+ amplitude_2*np.exp( - (a_2*((x-xo_2)**2) + 2*b_2*(x-xo_2)*(y-yo_2) 
                            + c_2*((y-yo_2)**2)))
    
    return g.ravel()


def determine_brighter(img, x, y, x2, y2, pix, redshift):
    kpc_arcmin=cosmo.kpc_proper_per_arcmin(redshift)#insert the redshift to get the kpc/arcmin scaling

    
    ap_size=(3*(kpc_arcmin.value/60))/pix ###3 arcsec diameter * (kpc/arcsec)   / (kpc/pix) -->
    #This is now in units of pixels
    
    '''step 1: define the circular aperture'''
    positions = [(x, y), (x2, y2)]
    apertures = CircularAperture(positions, ap_size)
    phot_table = aperture_photometry(img, apertures)
    total_light_1=phot_table['aperture_sum'][0]
    total_light_2=phot_table['aperture_sum'][1]

    
    
    masks = apertures.to_mask(method='center')
    mask = masks[0]

    image = mask.to_image(shape=((shape(img)[0], shape(img)[0])))
 
    
    return total_light_1, total_light_2


def clip_image(ins, pixelscale, redshift, xcen, ycen):
 
    kpc_arcmin=cosmo.kpc_proper_per_arcmin(redshift)#insert the redshift  
    #print(kpc_arcmin.value/60, 'kpc per arcsec')
    '''Divide the pixelscale (kpc/pix) by kpc/arcsec to get arcsec
    size of pixels'''
    size_a=pixelscale/(kpc_arcmin.value/60)

    '''lets make this image as big as possible'''
    min_pix_size = min([xcen,ycen,300-xcen,300-ycen])
    arc_size = int(min_pix_size*size_a)
    num_pix_half=int(arc_size/size_a)#was 25, arc_size
    '''50" per side'''
    
    
    print('Size of Cutout ~~~~'+str(arc_size))
    
    
    
    
    
 
    
    
    
    if xcen-num_pix_half < 0 or ycen-num_pix_half < 0 or xcen+num_pix_half > 300 or ycen+num_pix_half > 300 or num_pix_half==0:
        print('Outside of the box')
        clipped=0#(ins[xcen-num_pix_half:xcen+num_pix_half,ycen-num_pix_half:ycen+num_pix_half])#0
        tag='no'
    
    else:
        #build clipped and centered image
        clipped=(ins[xcen-num_pix_half:xcen+num_pix_half,ycen-num_pix_half:ycen+num_pix_half])
        tag='yes'

 
    return clipped, size_a, num_pix_half, tag, xcen, ycen, arc_size


'''Now I have to convert the units of LAURAS sims into nanomaggies and AB mags (mags of DR7 and DR13)
'''
def nanomags(z, pixscale, camera_data, view, number):

    c = 299792.458*1000#to get into m/s

 

    pixelscale=pixscale

    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    
    d_A = cosmo.comoving_distance(z).value/(1+z)
    
    '''Alternate way'''
    '''import cosmolopy.distance as cd
    cosmo = {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, 'h' :0.72}
    cosmo = cd.set_omega_k_0(cosmo)
    
    d_a = cd.angular_diameter_distance(z, **cosmo)
    print('angular diameter distance', d_a)'''
    
    #here's a good review of all the different distances cosmology distances:
    #http://www.astro.ufl.edu/~guzman/ast7939/projects/project01.html

    #Convert from specific intensity units (W/m/m^2/sr) from SUNRISE to Janskies (W/Hz/m^2): 
    Janskies=np.array(10**(26)*camera_data*(pixelscale/(1000*d_A))**2*((6185.2*10**(-10))**2/c), dtype='>f4')#*np.pi*((6185.2*10**(-10))**2/c),
    #this 1.35e-6 comes from the arcsin(R_sky/Distance to object)
    #the answer needs to be in radians


    
    

    #J=10^-26 W/m^2/Hz, so units of flux density
    #reference site: http://www.cv.nrao.edu/course/astr534/Brightness.html
    #We need to go from a spectral brightness (I_nu) which is in m units
    #To a flux density (S_nu) which is in units of Janskies (W/m^2/Hz)

    #So you need to multiply the Fν by c / λ^2 to convert it into Fλ. 
    #But we are not done yet! Recalling from above, the units of Fλ 
    #are not an energy density. You need to get another factor of λ 
    #in there to make the units work out to be energy density: 
    #calculate λFλ to get units of ergs/s/cm^2.

    
    #zero-point flux density is 3631 Jy

 #*15.8#/(1+z)**5
 #Then convert from this to a magnitude: m = [22.5 mag] − 2.5 log 10 f
    

    m = - 2.5*np.log10(abs(Janskies) / 3631)
    

    max_mag = m.min()

    

    
    
    
    '''Lets say I want everything to be 20th mag in order to test if this is the same result as everything varying'''
    scaling = m.min()-20
    
    Janskies_bright = Janskies*100**(1/5)

    m =  - 2.5*np.log10(abs(Janskies_bright) / 3631)

    m_asinh = (-2.5/math.log(10))*(np.arcsinh((Janskies_bright/3631)/(2*1.2*10**(-10))) + math.log(1.2*10**(-10)))
    
    
 
    plt.clf()
    fig=plt.figure()
    ax1=fig.add_subplot(211)
    
    im1=ax1.imshow(m,cmap='afmhot_r')#,norm=matplotlib.colors.LogNorm()
    #plt.title(str(m.min()))
    #ax1.axis('off')
        
    ax1.annotate(r't = '+str(round(myr_actual,2))+' Gyr', xy=(0.5,0.9), color='black', xycoords='axes fraction', size=15)
    plt.title(str(round(m.min(),2)))
    plt.colorbar(im1, label='r-band apparent magnitude')

    ax2 = fig.add_subplot(212)
    im2=ax2.imshow(m_asinh,cmap='afmhot_r')#,
    plt.title(str(round(m_asinh.min(),2)))
    plt.colorbar(im2, label='r-band apparent magnitude') 
    plt.savefig('../MaNGA_Papers/Paper_I/app_mags_'+str(view)+'_'+str(number)+'.png')
    
    
    nanomaggy=(Janskies_bright/(3.631*10**(-6)))
    
    '''Now convert into the correct background:)'''

    #nanomaggies and stuff: (Finkbeiner et al. 2004)

    '''The average background and std as well as the readnoise and gain
    are from another code of mine:
        SDSS_noise_imaging/SDSS_r_imaging.ipynb'''

    #first, convert to counts (dn) using the average value of cimg from the SDSS frame images
    #dn=img/cimg+simg
    cimg=0.005005225 #nanomaggies/count
    gain=4.735
    darkvar=1.1966
    simg=121.19590411 #average background value (counts) pre background subraction (used to calculate poisson error)
    counts=(nanomaggy)/cimg+simg
    sigma_counts=np.sqrt(counts/gain+darkvar)
    sigma_nanomags=sigma_counts*cimg
    
    nanomag_bg = counts*cimg




    



    '''The sky resids are given by:''' 
    sky_resids_mine=cimg*np.random.normal(0.331132,5.63218,shape(nanomaggy))
    sky_resids_mine_counts=np.random.normal(0.331132,5.63218,shape(nanomaggy))
    d_image=(nanomaggy)#/10#+sky_resids_mine
    degraded_image=d_image
    degraded_image_counts=d_image/cimg#will use this one in the future
    

    
    
    #return n, sky resids added (nanomags), poisson noise, resids in nanomags, sky resids added (counts), poisson noise, resids in counts
    return nanomaggy, degraded_image, sigma_nanomags, sky_resids_mine, degraded_image_counts, sigma_counts, sky_resids_mine_counts, m.min()




def convolve_rebin_image(number, z, pixscale, view, counts, t_exp, size):#all of these are in nanomags
    #PSF = 1.61 arcsec 
    

    

    kpc_arcmin=cosmo.kpc_proper_per_arcmin(z)#insert the redshift to get the kpc/arcmin scaling

    sigma=1.43/2.355#apparently the preimage sigma is not large :) was 1.61
    ##kpc/pix is the pixelscale
    ##conversion factor is kpc/"
    #1.61 is the FWHM in arcsec of the psf
    #pixelscale is kpc/pix
    kernel_sigma_pix=(sigma*(kpc_arcmin.value/60))/pixscale
    '''(arcsec * kpc/arcsec) / (kpc/pix) --> pixels'''
  

    gaussian_2D_kernel = Gaussian2DKernel(kernel_sigma_pix)#standard deviation in pixels


    #result = convolve(np.sum(CAMERA0.data[:,:,:],axis=0), gaussian_2D_kernel)
    result = (convolve(counts, gaussian_2D_kernel))

    counts_OG = counts
    try:
        rebin = scipy.ndimage.zoom(result, 1/(np.shape(result)[0]/(size*2/0.396)), order=0)
    except ZeroDivisionError:
        plt.clf()
        plt.imshow(counts)
        plt.title('Size '+str(size))
        plt.savefig('troubleshoot_rebin.pdf')
        STOP
    '''now pad it with zeros'''
    factor = (50-np.shape(rebin)[0]/2*0.396)/0.396
    padded = np.pad(rebin, 0, 'constant', constant_values=0)#int(factor)
    
    sky_resids_mine_counts=np.random.normal(0.331132,5.63218,shape(padded))#0.331132

    
    indent = 0#int(factor)#
    '''Try to cut down sky_resids_mine_counts'''
    sky_resids_mine_counts_cutout=sky_resids_mine_counts[indent:indent+np.shape(rebin)[0],indent:indent+np.shape(rebin)[0]]
    
    '''plt.clf()
    fig=plt.figure()
    ax1=fig.add_subplot(211)
    im1=ax1.imshow(rebin, norm=matplotlib.colors.LogNorm())
    plt.colorbar(im1)

    ax2=fig.add_subplot(212)
    im2=ax2.imshow(abs(padded+sky_resids_mine_counts), norm=matplotlib.colors.LogNorm())
    plt.colorbar(im2)
    plt.savefig('padded.pdf')'''
    
    
   
    gain=4.735
    darkvar=1.1966
    simg=121.19590411
    counts = padded + simg #+ sky_resids_mine
    sigma_counts=np.sqrt(counts/gain+darkvar)
    sigma_counts_cutout = np.sqrt((rebin+simg)/gain+darkvar)



    import seaborn as sns
    sns.set_style('dark')
    plt.clf()
    fig, axes = plt.subplots(ncols=4,nrows=2, figsize=(10,4))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    #ax = fig.add_subplot(2,4,1)
    masked = np.ma.masked_where(abs(counts_OG) < 10**0, abs(counts_OG))
    
    
    im = axes.flat[0].imshow(masked.filled(fill_value=1), cmap='afmhot', norm=matplotlib.colors.LogNorm(vmin=10**0, vmax=10**4))
    axes.flat[0].set_title(r'Clip Simulated Image', size=10)
    axes.flat[0].annotate('1', size=17, color='white', xy=(0.03,0.84), xycoords='axes fraction')
    
    #ax0 = fig.add_subplot(2,4,2)
    im0 = axes.flat[1].imshow(abs(rebin), cmap='afmhot', norm=matplotlib.colors.LogNorm(vmin=10**0, vmax=10**4))
    axes.flat[1].set_title('Convolve and Rebin', size=10)
    axes.flat[1].annotate('2', size=17, color='white', xy=(0.03,0.84), xycoords='axes fraction')
    #plt.colorbar(im0)
    
    sky_resids_mine_counts = np.random.normal(0.331132,5.63218,shape(rebin))#0.331132
    
    #ax1 = fig.add_subplot(2,4,3)
    im1 = axes.flat[2].imshow(abs(sky_resids_mine_counts), cmap='afmhot', norm=matplotlib.colors.LogNorm(vmin=10**0, vmax=10**4))
    axes.flat[2].set_title('Add Residual Background Noise', size=10)
    axes.flat[2].annotate('3', size=17, color='white', xy=(0.03,0.84), xycoords='axes fraction')
    #plt.colorbar(im1)
    
    
    #sky_resids_mine_counts = np.random.normal(0.331132,5.63218,shape(rebin))

    
    
    #ax3 = fig.add_subplot(2,4,4)
    im3 = axes.flat[3].imshow(abs(sky_resids_mine_counts+rebin), cmap='afmhot', norm=matplotlib.colors.LogNorm(vmin=10**0, vmax=10**4))
    axes.flat[3].set_title('Mock Image', size=10)
    #axes.flat[3].annotate('4',size=17, color='white', xy=(0.03,0.84), xycoords='axes fraction')
    #plt.colorbar(im3)
    


    

    SDSS_image = pyfits.open('../MergerMonger/imaging/pet_radius_587727225690194001.fits')
    #587727225690194001

    im4 = axes.flat[7].imshow(SDSS_image[0].data, cmap='afmhot', norm=matplotlib.colors.LogNorm(vmin=10**(0), vmax=10**4))
    axes.flat[7].set_title('SDSS Image', size=10)
    #axes.flat[7].annotate('E', size=17, color='white', xy=(0.03,0.84), xycoords='axes fraction')

    locs = axes.flat[7].get_xticks()
    print('locs', locs)
    labels = axes.flat[7].get_xticklabels()
    new_labels = [(x-max(locs)/2)*0.396 for x in locs]
 
    xlocs_pix = [0,np.shape(rebin)[0]/2,np.shape(rebin)[0]]
    xlocs= [int(round((x-np.shape(rebin)[0]/2)*0.396,1)) for x in xlocs_pix]
    

    axes.flat[3].set_xticks(xlocs_pix)
    axes.flat[3].set_xticklabels(xlocs, size=10)
    axes.flat[3].set_yticks(xlocs_pix)
    axes.flat[3].set_yticklabels(xlocs, size=10)

    xlocs_pix = [0,np.shape(SDSS_image[0].data)[0]/2,np.shape(SDSS_image[0].data)[0]]
    xlocs= [int(round((x-np.shape(SDSS_image[0].data)[0]/2)*0.396,0)) for x in xlocs_pix]

    axes.flat[7].set_xticks(xlocs_pix)
    axes.flat[7].set_xticklabels(xlocs, size=10)
    axes.flat[7].set_yticks(xlocs_pix)
    axes.flat[7].set_yticklabels(xlocs, size=10)

    
    
    #plt.colorbar(im4)

    axes.flat[4].axis('off')

    axes.flat[5].axis('off')

    axes.flat[6].axis('off')

    axes.flat[0].axis('off')

    axes.flat[1].axis('off')

    axes.flat[2].axis('off')

    

    #axes.flat[3].set_ylabel('Spatial Position [$^{\prime\prime}$]', size=15)
    #axes.flat[3].set_xlabel('Spatial Position [$^{\prime\prime}$]', size=15)
    axes.flat[7].set_xlabel('Spatial Position [$^{\prime\prime}$]', size=10)
    axes.flat[7].set_ylabel('Spatial Position [$^{\prime\prime}$]', size=10)

#cbar = fig.colorbar(im, cax=cax).set_label(label='Counts', size=15)#, labelsize=15)#, size=15)#, font=20)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85,0.15,0.02,0.7])#[left, bottom, width, height] 
    cbar2=fig.colorbar(im, cax=cbar_ax)#, aspect=20)#, labelsize=15)#, size=15)#, font=20)

    cbar2.ax.tick_params(labelsize=10)
    cbar2.set_label('Counts', size=10)
    plt.savefig('mock_images_rebin.pdf')






    plt.clf()
    fig, axes = plt.subplots(ncols=3,nrows=1, figsize=(9,3))


    #ax = fig.add_subplot(2,4,1)
    masked = np.ma.masked_where(abs(counts_OG) < 10**0, abs(counts_OG))
    
    
    im = axes.flat[0].imshow(masked.filled(fill_value=1), cmap='afmhot', norm=matplotlib.colors.LogNorm(vmin=10**0, vmax=10**4))
    axes.flat[0].set_title(r'Simulated Image', size=17, color='white')
    #axes.flat[0].annotate('Simulated Image', size=12, color='white', xy=(0.05,0.9), xycoords='axes fraction')
    
    im1 = axes.flat[1].imshow(abs(sky_resids_mine_counts+rebin), cmap='afmhot', norm=matplotlib.colors.LogNorm(vmin=10**0, vmax=10**4))
    axes.flat[1].set_title('Mock SDSS Image', size=17, color='white')
        #axes.flat[1].annotate('Mock SDSS Image', size=12, color='white', xy=(0.05,0.9), xycoords='axes fraction')
    #axes.flat[3].annotate('4',size=17, color='white', xy=(0.03,0.84), xycoords='axes fraction')
    #plt.colorbar(im3)
    


    

    SDSS_image = pyfits.open('../MergerMonger/imaging/pet_radius_587728669878845746.fits')##587727225690194001
    #587727225690194001

    im2 = axes.flat[2].imshow(SDSS_image[0].data, cmap='afmhot', norm=matplotlib.colors.LogNorm(vmin=10**(0), vmax=10**4))
    axes.flat[2].set_title('Actual SDSS Image', size=17, color='white')
    #axes.flat[2].annotate('SDSS Image', size=12, color='white', xy=(0.05,0.9), xycoords='axes fraction')

    axes.flat[0].axis('off')

    axes.flat[1].axis('off')

    axes.flat[2].axis('off')
    #plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.rcParams['axes.facecolor']='black'
    plt.rcParams['savefig.facecolor']='black'
    plt.savefig('three.png')
    
    

    

    '''This prepares the files used by galfit - an error image and an input image both in counts'''
    outfile = 'GALFIT_folder/out_convolved_'+str(view)+'_'+str(number)+'.fits'
    hdu = fits.PrimaryHDU(np.flipud((sky_resids_mine_counts+padded)*t_exp))
    hdu_number = 0
    hdu.writeto(outfile, overwrite=True)
    hdr=fits.getheader(outfile, hdu_number)
    
    hdr['EXPTIME'] = 1
    hdr['EXPTIME']
    
    hdu.writeto(outfile, overwrite=True)

    outfile = 'GALFIT_folder/out_sigma_convolved_'+str(view)+'_'+str(number)+'.fits'
    hdu = fits.PrimaryHDU(np.flipud((sigma_counts)*t_exp))
    hdu_number = 0
    hdu.writeto(outfile, overwrite=True)
    hdr=fits.getheader(outfile, hdu_number)
    
    hdr['EXPTIME'] = 1
    hdr['EXPTIME']
    hdu.writeto(outfile, overwrite=True)
    
    
    
    
    return padded + sky_resids_mine_counts, sigma_counts, rebin+sky_resids_mine_counts_cutout, sigma_counts_cutout

    

def convolve_image(number,nanomaggy,degraded_image,sigma_nanomags,z,pixscale, view, background, counts, counts_sig, background_counts, t_exp):#all of these are in nanomags
    #PSF = 1.61 arcsec
    

    

    kpc_arcmin=cosmo.kpc_proper_per_arcmin(z)#insert the redshift to get the kpc/arcmin scaling

    sigma=1.61/2.355#apparently the preimage sigma is not large :)
    ##kpc/pix is the pixelscale
    ##conversion factor is kpc/"
    #1.61 is the FWHM in arcsec of the psf
    #pixelscale is kpc/pix
    kernel_sigma_pix=(sigma*(kpc_arcmin.value/60))/pixscale
    '''(arcsec * kpc/arcsec) / (kpc/pix) --> pixels'''
  

    gaussian_2D_kernel = Gaussian2DKernel(kernel_sigma_pix)#standard deviation in pixels



    #result = convolve(np.sum(CAMERA0.data[:,:,:],axis=0), gaussian_2D_kernel)
    result_nano = (convolve(nanomaggy, gaussian_2D_kernel))
    result = (convolve(degraded_image, gaussian_2D_kernel))
    result_bg = (convolve(background, gaussian_2D_kernel))
    result_bg_counts = (convolve(background_counts, gaussian_2D_kernel))
    result_error = (convolve(sigma_nanomags, gaussian_2D_kernel))
    
    result_counts = (convolve(counts, gaussian_2D_kernel))
    result_error_counts = (convolve(counts_sig, gaussian_2D_kernel))

   
    '''plt.clf()
    plt.imshow(result)
    plt.colorbar()
    plt.savefig('before.pdf')'''
    
    
   
    return result_counts


'''Prepares the default file for source extractor to read (I played with the parameters to the point where they work well:'''
def write_sex_default(view,num_feedme):
    os.chdir(os.path.expanduser('q0.5_fg0.3_allrx10'))
    file = open('default_'+str(view)+'_'+str(num_feedme)+'.sex', "w")
    file.write('# Default configuration file for SExtractor 2.5.0'+'\n')
        
    file.write('CATALOG_NAME     test_'+str(view)+'_'+str(num_feedme)+'.cat       # name of the output catalog'+'\n')
    file.write('CATALOG_TYPE     ASCII_HEAD     # '+'\n')
    file.write('PARAMETERS_NAME  default.param  # name of the file containing catalog contents'+'\n')

    file.write('#------------------------------- Extraction ----------------------------------'+'\n')

    file.write('DETECT_TYPE      CCD            # CCD (linear) or PHOTO (with gamma correction)'+'\n')
    file.write('DETECT_MINAREA   2            # minimum number of pixels above threshold'+'\n')
    file.write('DETECT_THRESH    1.5            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2'+'\n')
    file.write('ANALYSIS_THRESH  2            # <sigmas> or <threshold>,<ZP> in mag.arcsec-2'+'\n')

    file.write('FILTER           Y              # apply filter for detection (Y or N)?'+'\n')
    file.write('FILTER_NAME      default.conv   # name of the file containing the filter'+'\n')

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
    file.write('STARNNW_NAME     default.nnw    # Neural-Network_Weight table filename'+'\n')

    file.write('#------------------------------ Background -----------------------------------'+'\n')

    file.write('BACK_SIZE        64             # Background mesh: <size> or <width>,<height>'+'\n')
    file.write('BACK_FILTERSIZE  3              # Background filter: <size> or <width>,<height>'+'\n')
    file.write('BACKPHOTO_TYPE   GLOBAL         # can be GLOBAL or LOCAL'+'\n')

    file.write('#------------------------------ Check Image ----------------------------------'+'\n')

    file.write('CHECKIMAGE_TYPE  APERTURES          # can be NONE, BACKGROUND, BACKGROUND_RMS,'+'\n')
                                # MINIBACKGROUND, MINIBACK_RMS, -BACKGROUND,
                                # FILTERED, OBJECTS, -OBJECTS, SEGMENTATION,
                                # or APERTURES
    file.write('CHECKIMAGE_NAME  aps_'+str(view)+'_'+str(num_feedme)+'.fits     # Filename for the check-image'+'\n')

    file.write('#--------------------- Memory (change with caution!) -------------------------'+'\n')

    file.write('MEMORY_OBJSTACK  3000           # number of objects in stack'+'\n')
    file.write('MEMORY_PIXSTACK  300000         # number of pixels in stack'+'\n')
    file.write('MEMORY_BUFSIZE   1024           # number of lines in buffer'+'\n')

    file.write('#----------------------------- Miscellaneous ---------------------------------'+'\n')

    file.write('VERBOSE_TYPE     QUIET        # can be QUIET, NORMAL or FULL'+'\n')
    file.write('WRITE_XML        Y              # Write XML file (Y/N)?'+'\n')
    file.write('XML_NAME         sex.xml        # Filename for XML output'+'\n')


    file.close()

'''Runs source extractor from within python'''
    
def run_sex(view,num_feedme):
    
    os.system("sex -c default_"+str(view)+"_"+str(num_feedme)+".sex "+"pet_radius_"+str(view)+"_"+str(num_feedme)+".fits")
    

'''Extracts parameters from source extractor that are useful as a Galfit input/ first guess'''

def sex_params_galfit(view,num_feedme):
    #this one also determines how many bulges there are
    #os.chdir(os.path.expanduser('q0.5_fg0.3_allrx10'))
    #   1 NUMBER                 Running object number
    #   2 X_IMAGE                Object position along x                                    [pixel]
    #   3 Y_IMAGE                Object position along y                                    [pixel]
    #   4 A_IMAGE                Profile RMS along major axis                               [pixel]
    #   5 B_IMAGE                Profile RMS along minor axis                               [pixel]
    #   6 FLUX_AUTO              Flux within a Kron-like elliptical aperture                [count]
    #   7 MAG_AUTO               Kron-like elliptical aperture magnitude                    [mag]
    #   8 FLUX_RADIUS            Fraction-of-light radii                                    [pixel]
    #   9 PETRO_RADIUS           Petrosian apertures in units of A or B
    #  10 KRON_RADIUS            Kron apertures in units of A or B
    #  11 FLUX_PETRO             Flux within a Petrosian-like elliptical aperture           [count]
    #  12 FLAGS                  Extraction flags
    #  13 THETA_IMAGE            Position angle (CCW)
    #  14 BACKGROUND

    file_path = 'test_'+str(view)+'_'+str(num_feedme)+'.cat'
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
            A.append(float(words[3]))
            B.append(float(words[4]))
            mag_auto.append(float(words[6]))
            eff_radius.append(float(words[7]))#this is the effective radius
            pet_mult.append(float(words[8]))#this is the petrosian radius
            B_mult.append(float(words[9]))
            flux_pet.append(float(words[10]))
            PA.append(float(words[12]))
            back.append(float(words[13]))
    try:
        max_i=flux_pet.index(max(flux_pet))
    except ValueError:
        return 0, 0
    x_max=x_pos[max_i]
    y_max=y_pos[max_i]
    sex_pet_r=(pet_mult[max_i]*A[max_i])
    minor=pet_mult[max_i]*B[max_i]
    flux=flux_pet[max_i]
    mag_max=flux_pet[max_i]
    eff_radius_1=eff_radius[max_i]
    B_A_1=B[max_i]/A[max_i]
    PA_1=PA[max_i]#was -90-PA[max_i]
    back_1=back[max_i]
    
    
    if len(x_pos)==1:
        n_bulges=1
        return sex_pet_r, minor, flux, x_max, y_max, mag_max, n_bulges, eff_radius_1, B_A_1, PA_1, back_1
    else:
        import heapq
        two_largest=heapq.nlargest(2, flux_pet)
        if two_largest[1] > 0.1*two_largest[0]:
            #then it's 1/10th of the brightness and is legit
            n_bulges=2
            max_i_sec=flux_pet.index(two_largest[1])
            x_max_2=x_pos[max_i_sec]
            y_max_2=y_pos[max_i_sec]
            mag_max_2=flux_pet[max_i_sec]
            eff_radius_2=eff_radius[max_i_sec]
            B_A_2=B[max_i_sec]/A[max_i_sec]
            PA_2=PA[max_i_sec] #was -90-PA[max_i]+180
            return sex_pet_r, minor, flux, x_max, y_max, mag_max, n_bulges, eff_radius_1, B_A_1, PA_1, back_1, x_max_2, y_max_2, mag_max_2, eff_radius_2, B_A_2, PA_2
        else:
            n_bulges=1
            return sex_pet_r, minor, flux, x_max, y_max, mag_max, n_bulges, eff_radius_1, B_A_1, PA_1, back_1


'''def sex_params_galfit(view,num_feedme):
    
    file_path = 'test_'+str(view)+'_'+str(num_feedme)+'.cat'
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
            return sex_pet_r, minor, flux, x_max, y_max, n_bulges, eff_radius_1, B_A_1, PA_1, back_1'''


def plot_sex(view,num_feedme,xpos,ypos):
    pic='aps_'+str(view)+'_'+str(num_feedme)+'.fits'
    plt.clf()
    im=pyfits.open(pic)

    
    #optional plotting of the Source Extractor output:


    '''plt.clf()
    
    plt.title('Sex Apertures')
    plt.imshow(im[0].data,norm=matplotlib.colors.LogNorm())
    plt.scatter(xpos,ypos, color='black')
    plt.savefig('sex_aps_'+str(view)+'_'+str(num_feedme)+'.pdf')'''
    return im[0].data#this is the bg image
 
    #os.chdir(os.path.expanduser('/Users/beckynevin/Documents/Backup_My_Book/My_Passport_backup/LAURA_SIMS'))


def write_galfit_feedme(view,number,xcen,ycen,xcen2,ycen2, mag, mag_zpt, num_bulges, length_gal, r_1, r_2, mag_2, B_A_1, PA_1, B_A_2, PA_2, background, z, pixscale):

    kpc_arcmin=cosmo.kpc_proper_per_arcmin(z)
    #pixelscale is kpc/pix
    arc_pix=1/((kpc_arcmin.value/60)/pixscale)
    #kpc/" divided by kpc/pix is pix/" and divide one more time
    
    os.chdir('/Users/beckynevin/Documents/Backup_My_Book/My_Passport_backup/LAURA_Sims/')
    if num_bulges==2: 
    
        '''I need to make a code to write out the GALFIT.feedme file'''
        file = open('GALFIT_folder/galfit.feedme_'+str(view)+'_'+str(number), "w")
        file.write('==============================================================================='+'\n')

        file.write('# IMAGE and GALFIT CONTROL PARAMETERS'+'\n')


        file.write('A) out_convolved_'+str(view)+'_'+str(number)+'.fits            # Input data image (FITS file)'+'\n')
        file.write('B) out_'+str(view)+'_'+str(number)+'.fits       # Output data image block'+'\n')
        file.write('C) out_sigma_convolved_'+str(view)+'_'+str(number)+'.fits                # Sigma image name (made from data i$'+'\n')
        file.write('D) none   #        # Input PSF image and (optional) diffusion kernel'+'\n')
        file.write('E) none                   # PSF fine sampling factor relative to data'+'\n')
        file.write('F) none                # Bad pixel mask (FITS image or ASCII coord list)'+'\n')
        file.write('G) none                # File with parameter constraints (ASCII file)'+'\n')
        file.write('H) 1    '+str(length_gal)+'   1    '+str(length_gal)+'   # Image region to fit (xmin xmax ymin ymax)'+'\n')
        file.write('I) '+str(length_gal)+' '+ str(length_gal)+'          # Size of the convolution box (x y)'+'\n')
        file.write('J) '+str(mag_zpt)+' # Magnitude photometric zeropoint'+'\n')
        file.write('K) '+str(arc_pix)+'  '+str(arc_pix)+'        # Plate scale (dx dy)    [arcsec per pixel]'+'\n')
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
        file = open('GALFIT_folder/galfit.feedme_'+str(view)+'_'+str(number), "w")
        file.write('==============================================================================='+'\n')

        file.write('# IMAGE and GALFIT CONTROL PARAMETERS'+'\n')


        file.write('A) out_convolved_'+str(view)+'_'+str(number)+'.fits            # Input data image (FITS file)'+'\n')
        file.write('B) out_'+str(view)+'_'+str(number)+'.fits       # Output data image block'+'\n')
        file.write('C) out_sigma_convolved_'+str(view)+'_'+str(number)+'.fits                # Sigma image name (made from data i$'+'\n')
        file.write('D) none   #        # Input PSF image and (optional) diffusion kernel'+'\n')
        file.write('E) none                   # PSF fine sampling factor relative to data'+'\n')
        file.write('F) none                # Bad pixel mask (FITS image or ASCII coord list)'+'\n')
        file.write('G) none                # File with parameter constraints (ASCII file)'+'\n')
        file.write('H) 1    '+str(length_gal)+'   1    '+str(length_gal)+'   # Image region to fit (xmin xmax ymin ymax)'+'\n')
        file.write('I) '+str(length_gal)+' '+ str(length_gal)+'          # Size of the convolution box (x y)'+'\n')
        file.write('J) '+str(mag_zpt)+' # Magnitude photometric zeropoint'+'\n')
        
        file.write('K) '+str(arc_pix)+'  '+str(arc_pix)+'       # Plate scale (dx dy)    [arcsec per pixel]'+'\n')
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

 
        
        file.write('# Object number: 3'+'\n')
        file.write(' 0) sky                    #  object type'+'\n')
        file.write(' 1) '+str(background)+'      1          #  sky background at center of fitting region [ADU$'+'\n')
        #was 100.3920      1                                                                                                                                                                           
        file.write(' 2) 0.0000      0          #  dsky/dx (sky gradient in x)'+'\n')
        file.write(' 3) 0.0000      0          #  dsky/dy (sky gradient in y)'+'\n')
        file.write(' Z) 0                      #  output option (0 = resid., 1 = Dont subtract)'+'\n')

        file.write('================================================================================'+'\n')


    file.close()

def run_galfit(view,num_feedme):
    os.chdir(os.path.expanduser('GALFIT_folder'))
    os.system("galfit galfit.feedme_"+str(view)+"_"+str(num_feedme)+">/dev/null 2>&1")
    os.chdir(os.path.expanduser('/Users/beckynevin/Documents/Backup_My_Book/My_Passport_backup/LAURA_SIMS'))

def galfit_params(view,output_number,num_bulges,kpcpix):
    
    '''First, figure out how many bulges'''
 
    #plt.clf()
    try:
        output='GALFIT_folder/out_'+str(view)+'_'+str(output_number)+'.fits'
        out=pyfits.open(output)
    except FileNotFoundError:
        return None
    
    #fig=plt.figure()
    #ax1=fig.add_subplot(211)
    #im1=ax1.imshow(out[1].data, norm=matplotlib.colors.LogNorm())
    #plt.colorbar(im1,label='Magnitudes')
    #ax2=fig.add_subplot(212)
    #im2=ax2.imshow(out[2].data,norm=matplotlib.colors.LogNorm())
    #plt.colorbar(im2,label='Magnitudes')
    #plt.savefig('GALFIT_folder/side_by_side_galfit_input_'+str(view)+'_'+str(output_number)+'.pdf')
    
    
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
        AR1=float(out[2].header['1_AR'][:7])
        try:
            sersic = float(out[2].header['1_N'][:7])
        except ValueError or KeyError:
            sersic=float(out[2].header['1_N'][1:7])
        

        
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
                       
        

    galfit_input=out[1].data
    galfit_model=out[2].data
    galfit_resid=out[3].data
    
    
    chi2=(out[2].header['CHI2NU'])
    
    
    return sep, flux_ratio, PA1, PA2, x_1, y_1, x_2, y_2, AR1, out[2].data, sersic



'''Uses the Petrosian radius measured with Source Extractor and the semimajor axis and the petrosian flux
to calculate the surface brightness at the petrosian radius which we then will use to construct the segmentation
maps in a later step'''
def input_petrosian_radius(input_r, input_b, flux_w_i):
    
    Area=(input_b*input_r*np.pi) #area in pixels of an ellipse
    limiting_sb=0.2*(flux_w_i/Area)#was 0.2*(flux_w_i/Area)
    
    
    return limiting_sb#returns the surface brightness at the petrosian radius



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
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)

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
            'Image min = %.4f' % (m) + '\n' +
            'Image max = %.4f' % (M))
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


def asymmetry(gal_zoom, gal_zoom_bg):
    A_img=np.sum(abs(gal_zoom-np.rot90(np.rot90(gal_zoom))))/(np.sum(abs(gal_zoom)))#muliply bottom by 2?
    #plt.clf()
    #plt.imshow(abs(gal_zoom-np.rot90(np.rot90(gal_zoom))))
    #plt.annotate('A_S = '+str(round(A,2)), xy=(0.1,0.1), xycoords='axes fraction', color='white')
    #plt.savefig('../MaNGA_Papers/Paper_I/Shape_Asymmetry_'+str(myr[i])+'_'+str(viewpt)+'_'+str(merger_sim[i])+'.png',bbox_inches='tight')
    A_bg = np.sum(abs(gal_zoom_bg-np.rot90(np.rot90(gal_zoom_bg))))/(np.sum(abs(gal_zoom)))

    A = A_img - A_bg
        
    return A
 


'''The simulated images come in packages with 7 viewing angles and 7 snapshots each.

Sometimes the images have different pixelscales because the earlier snapshots have to be more 
zoomed out, so read this from the header, ['CD1_1'].

Also, the point in time does not exactly correspond to the time given in the name.'''
















img_list=['q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_005.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_060.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_090.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_120.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_150.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_180.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_210.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_240.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_270.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_300.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_320.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_340.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_360.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_380.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_400.fits',
         'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_420.fits',
         'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_440.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_460.fits',
         'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_480.fits',
         'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_500.fits']



myr=[5,60,90,120,150,180,210,240,270,300,320,340,360,380,400,420,440,460,480,500]
merger=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
merger_sim=['fg3_m_15','fg3_m_15','fg3_m_15','fg3_m_15','fg3_m_15',
            'fg3_m_15','fg3_m_15','fg3_m_15','fg3_m_15','fg3_m_15',
            'fg3_m_15','fg3_m_15','fg3_m_15','fg3_m_15',
            'fg3_m_15','fg3_m_15','fg3_m_15','fg3_m_15','fg3_m_15','fg3_m_15']











img_list=['q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_170.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_180.fits',
    'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_185.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_190.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_195.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_205.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_210.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_220.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_225.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_230.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_240.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_250.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_260.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_265.fits',
          'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_275.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_285.fits',
          'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_295.fits',
            'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_305.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_311.fits',
        'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_315.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_320.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_005.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_010.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_020.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_030.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_040.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_060.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_080.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_100.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_120.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_140.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_160.fits']
myr=[170,180,185, 190, 195, 205, 210, 220, 225, 230, 240, 250, 260,265,275,285, 295,305,311, 315,320,
     5,10,20,30,40,60,80,100,120,140,160]
merger=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,
       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
merger_sim=['fg3_m12','fg3_m12','fg3_m12','fg3_m12',
            'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
            'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
            'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12']

















#210 is nice











img_list=['q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_170.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_180.fits',
    'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_185.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_190.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_195.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_205.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_210.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_220.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_225.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_230.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_240.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_250.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_260.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_265.fits',
          'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_275.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_285.fits',
          'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_295.fits',
            'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_305.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_311.fits',
        'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_315.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_320.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_005.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_010.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_020.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_030.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_040.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_060.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_080.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_100.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_120.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_140.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_160.fits']
myr=[170,180,185, 190, 195, 205, 210, 220, 225, 230, 240, 250, 260,265,275,285, 295,305,311, 315,320,
     5,10,20,30,40,60,80,100,120,140,160]
merger=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,
       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
merger_sim=['fg3_m12','fg3_m12','fg3_m12','fg3_m12',
            'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
            'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
            'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12']

img_list=['q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_320.fits']
merger=[1]
myr=[320]
merger_sim=['fg3_m15']







img_list=['q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_010.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_040.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_050.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_060.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_070.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_090.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_120.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_130.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_140.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_145.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_150.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_155.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_170.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_175.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_180.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_185.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_190.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_195.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_200.fits',
          'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_205.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_210.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_215.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_220.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_225.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_230.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_235.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_240.fits',
          'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_245.fits',
          'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_250.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_260.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_270.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_280.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_290.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_300.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_310.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_320.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_330.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_340.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_350.fits',
         'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_410.fits']
myr=[10,40,50,60,70,90,120,130,140,
    145,150,155,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,260,270,280,290,300,310,
    320,330,340,350,410]
merger=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1]
merger_sim=['fg1_m_13','fg1_m_13','fg1_m_13',
            'fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13',
           'fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13',
           'fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13']




img_list=['q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_170.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_180.fits',
    'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_185.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_190.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_195.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_205.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_210.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_220.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_225.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_230.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_240.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_250.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_260.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_265.fits',
          'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_275.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_285.fits',
          'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_295.fits',
            'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_305.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_311.fits',
        'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_315.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_320.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_005.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_010.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_020.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_030.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_040.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_060.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_080.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_100.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_120.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_140.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_160.fits']
myr=[170,180,185, 190, 195, 205, 210, 220, 225, 230, 240, 250, 260,265,275,285, 295,305,311, 315,320,
     5,10,20,30,40,60,80,100,120,140,160]
merger=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,
       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
merger_sim=['fg3_m12','fg3_m12','fg3_m12','fg3_m12',
            'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
            'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
            'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12']


















img_list=['q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_010.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_040.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_050.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_060.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_070.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_090.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_120.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_130.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_140.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_145.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_150.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_155.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_170.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_175.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_180.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_185.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_190.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_195.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_200.fits',
          'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_205.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_210.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_215.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_220.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_225.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_230.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_235.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_240.fits',
          'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_245.fits',
          'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_250.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_260.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_270.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_280.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_290.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_300.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_310.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_320.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_330.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_340.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_350.fits',
          'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_360.fits',
          'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_370.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_390.fits',
          'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_400.fits',
         'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_410.fits']
myr=[10,40,50,60,70,90,120,130,140,
    145,150,155,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,260,270,280,290,300,310,
    320,330,340,350,360,370,390,400,410]
merger=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1]
merger_sim=['fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13',
            'fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13',
           'fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13',
           'fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13']



















img_list=['isolated_galaxies/m0.5_fg0.3/broadband_005.fits','isolated_galaxies/m0.5_fg0.3/broadband_010.fits','isolated_galaxies/m0.5_fg0.3/broadband_030.fits',
          'isolated_galaxies/m0.5_fg0.3/broadband_050.fits','isolated_galaxies/m0.5_fg0.3/broadband_200.fits',
         'isolated_galaxies/m0.333_fg0.1/broadband_005.fits','isolated_galaxies/m0.333_fg0.1/broadband_010.fits','isolated_galaxies/m0.333_fg0.1/broadband_030.fits',
          'isolated_galaxies/m0.333_fg0.1/broadband_050.fits','isolated_galaxies/m0.333_fg0.1/broadband_100.fits','isolated_galaxies/m0.333_fg0.1/broadband_200.fits',
         'isolated_galaxies/m1_fg0.1/broadband_005.fits','isolated_galaxies/m1_fg0.1/broadband_010.fits','isolated_galaxies/m1_fg0.1/broadband_020.fits',
          'isolated_galaxies/m1_fg0.1/broadband_030.fits','isolated_galaxies/m1_fg0.1/broadband_040.fits','isolated_galaxies/m1_fg0.1/broadband_050.fits',
          'isolated_galaxies/m1_fg0.1/broadband_060.fits','isolated_galaxies/m1_fg0.1/broadband_100.fits','isolated_galaxies/m1_fg0.1/broadband_200.fits',
         'isolated_galaxies/m1_fg0.3/broadband_005.fits','isolated_galaxies/m1_fg0.3/broadband_010.fits','isolated_galaxies/m1_fg0.3/broadband_020.fits',
          'isolated_galaxies/m1_fg0.3/broadband_030.fits','isolated_galaxies/m1_fg0.3/broadband_040.fits','isolated_galaxies/m1_fg0.3/broadband_050.fits',
          'isolated_galaxies/m1_fg0.3/broadband_060.fits','isolated_galaxies/m1_fg0.3/broadband_100.fits','isolated_galaxies/m1_fg0.3/broadband_200.fits',
         'isolated_galaxies/m1_fg0.3_BT0.2/broadband_005.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_030.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_060.fits',
          'isolated_galaxies/m1_fg0.3_BT0.2/broadband_090.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_100.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_120.fits',
          'isolated_galaxies/m1_fg0.3_BT0.2/broadband_150.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_180.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_200.fits']
myr=[5,10,30,50,200,
    5,10,30,50,100,200,
    5,10,20,30,40,50,60,100,200,
    5,10,20,30,40,50,60,100,200,
    5,30,60,90,100,120,150,180,200]
merger_sim=['iso_m0.5_fg0.3','iso_m0.5_fg0.3','iso_m0.5_fg0.3','iso_m0.5_fg0.3','iso_m0.5_fg0.3',
           'iso_m0.333_fg0.1','iso_m0.333_fg0.1','iso_m0.333_fg0.1','iso_m0.333_fg0.1','iso_m0.333_fg0.1','iso_m0.333_fg0.1',
           'iso_m1_fg0.1','iso_m1_fg0.1','iso_m1_fg0.1','iso_m1_fg0.1','iso_m1_fg0.1','iso_m1_fg0.1','iso_m1_fg0.1','iso_m1_fg0.1','iso_m1_fg0.1',
           'iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3',
           'iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2']
merger=[0,0,0,0,0,0,
       0,0,0,0,0,0,0,
       0,0,0,0,0,0,0,0,0,0,
       0,0,0,0,0,0,0,0,0,0,
       0,0,0,0,0]








img_list=['isolated_galaxies/m0.5_fg0.3/broadband_005.fits','isolated_galaxies/m0.5_fg0.3/broadband_010.fits','isolated_galaxies/m0.5_fg0.3/broadband_030.fits',
          'isolated_galaxies/m0.5_fg0.3/broadband_050.fits','isolated_galaxies/m0.5_fg0.3/broadband_200.fits']
myr=[5,10,30,50,200]
merger_sim=['iso_m0.5_fg0.3','iso_m0.5_fg0.3','iso_m0.5_fg0.3','iso_m0.5_fg0.3','iso_m0.5_fg0.3']
merger=[0,0,0,0,0]








































img_list=['q0.333_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_020.fits',
    'q0.333_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_040.fits',
        'q0.333_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_070.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_080.fits',
         'q0.333_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_100.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_120.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_140.fits',
         'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_150.fits',
         'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_170.fits',
         'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_180.fits',
         'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_190.fits',
         'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_210.fits',
         'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_220.fits',
         'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_230.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_235.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_237.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_240.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_242.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_245.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_247.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_250.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_252.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_255.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_257.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_262.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_265.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_270.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_275.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_280.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_285.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_287.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_288.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_289.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_290.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_295.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_300.fits',
          'q0.333_fg0.3_allrx10_sunruns/hires_kin_late/broadband_305.fits',
          'isolated_galaxies/m1_fg0.3/broadband_005.fits','isolated_galaxies/m1_fg0.3/broadband_010.fits','isolated_galaxies/m1_fg0.3/broadband_020.fits',
          'isolated_galaxies/m1_fg0.3/broadband_030.fits','isolated_galaxies/m1_fg0.3/broadband_040.fits','isolated_galaxies/m1_fg0.3/broadband_050.fits',
          'isolated_galaxies/m1_fg0.3/broadband_060.fits','isolated_galaxies/m1_fg0.3/broadband_100.fits','isolated_galaxies/m1_fg0.3/broadband_200.fits']

myr=[20,40,70,80,100,120,140,150,170,180,190,210,220,230,235,237,
     240,242,245,247,250,252,255,257,262,265,270,275,280,285,
     287,288,289,290,295,300,305,
     5,10,20,30,40,50,60,100,200]
merger=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        0,0,0,0,0,0,0,0,0]
merger_sim=['fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13',
            'fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13',
            'fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13',
            'fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13','fg3_m_13',
            'iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3']




'''img_list=['isolated_galaxies/m1_fg0.3_BT0.2/broadband_005.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_030.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_060.fits',
          'isolated_galaxies/m1_fg0.3_BT0.2/broadband_090.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_100.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_120.fits',
          'isolated_galaxies/m1_fg0.3_BT0.2/broadband_150.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_180.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_200.fits']
myr=[5,30,60,90,100,120,150,180,200]
merger_sim=['iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2']
merger=[0,0,0,0,
       0,0,0,0,0]'''



img_list=['isolated_galaxies/m0.5_fg0.3/broadband_005.fits','isolated_galaxies/m0.5_fg0.3/broadband_010.fits','isolated_galaxies/m0.5_fg0.3/broadband_030.fits',
          'isolated_galaxies/m0.5_fg0.3/broadband_050.fits','isolated_galaxies/m0.5_fg0.3/broadband_200.fits',
         'isolated_galaxies/m1_fg0.3/broadband_005.fits','isolated_galaxies/m1_fg0.3/broadband_010.fits','isolated_galaxies/m1_fg0.3/broadband_020.fits',
          'isolated_galaxies/m1_fg0.3/broadband_030.fits','isolated_galaxies/m1_fg0.3/broadband_040.fits','isolated_galaxies/m1_fg0.3/broadband_050.fits',
          'isolated_galaxies/m1_fg0.3/broadband_060.fits','isolated_galaxies/m1_fg0.3/broadband_100.fits','isolated_galaxies/m1_fg0.3/broadband_200.fits',
         'isolated_galaxies/m1_fg0.3_BT0.2/broadband_005.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_030.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_060.fits',
          'isolated_galaxies/m1_fg0.3_BT0.2/broadband_090.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_100.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_120.fits',
          'isolated_galaxies/m1_fg0.3_BT0.2/broadband_150.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_180.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_200.fits']
myr=[5,10,30,50,200,
    5,10,20,30,40,50,60,100,200,
    5,30,60,90,100,120,150,180,200]
merger_sim=['iso_m0.5_fg0.3','iso_m0.5_fg0.3','iso_m0.5_fg0.3','iso_m0.5_fg0.3','iso_m0.5_fg0.3',
           'iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3',
           'iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2']
merger=[0,0,0,0,0,0,
     0,0,0,0,0,0,0,0,0,0,
       0,0,0,0,0,0,0]

img_list=['q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_170.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_180.fits',
    'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_185.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_190.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_195.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_205.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_210.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_220.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_225.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_230.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_240.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_250.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_260.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_265.fits',
          'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_275.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_285.fits',
          'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_295.fits',
            'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_305.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_311.fits',
        'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_315.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_320.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_005.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_010.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_020.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_030.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_040.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_060.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_080.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_100.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_120.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_140.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_160.fits']
myr=[170,180,185, 190, 195, 205, 210, 220, 225, 230, 240, 250, 260,265,275,285, 295,305,311, 315,320,
     5,10,20,30,40,60,80,100,120,140,160]
merger=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,
       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
merger_sim=['fg3_m12','fg3_m12','fg3_m12','fg3_m12',
            'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
            'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
            'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12']

img_list=['q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_005.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_060.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_090.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_120.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_150.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_180.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_210.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_240.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_270.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_300.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_320.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_340.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_360.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_380.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_400.fits',
         'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_420.fits',
         'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_440.fits',
          'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_460.fits',
         'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_480.fits',
         'q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_500.fits']



myr=[5,60,90,120,150,180,210,240,270,300,320,340,360,380,400,420,440,460,480,500]
merger=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
merger_sim=['fg3_m_15','fg3_m_15','fg3_m_15','fg3_m_15','fg3_m_15',
            'fg3_m_15','fg3_m_15','fg3_m_15','fg3_m_15','fg3_m_15',
            'fg3_m_15','fg3_m_15','fg3_m_15','fg3_m_15',
            'fg3_m_15','fg3_m_15','fg3_m_15','fg3_m_15','fg3_m_15','fg3_m_15']


img_list=['q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_010.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_040.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_050.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_060.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_070.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_090.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_120.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_130.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_early_cen1/broadband_140.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_145.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_150.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_155.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_170.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_175.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_180.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_185.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_190.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_195.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_200.fits',
          'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_205.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_210.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_215.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_220.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_225.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_230.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_235.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_240.fits',
          'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_245.fits',
          'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_250.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_260.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_270.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_280.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_290.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_300.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_310.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_320.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_330.fits',
    'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_340.fits','q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_sdss_z0.03_350.fits',
         'q0.333_fg0.1_allrx10_sunruns/hires_kin_late/broadband_410.fits',
          'isolated_galaxies/m0.333_fg0.1/broadband_005.fits','isolated_galaxies/m0.333_fg0.1/broadband_010.fits','isolated_galaxies/m0.333_fg0.1/broadband_030.fits',
          'isolated_galaxies/m0.333_fg0.1/broadband_050.fits','isolated_galaxies/m0.333_fg0.1/broadband_100.fits','isolated_galaxies/m0.333_fg0.1/broadband_200.fits']
myr=[10,40,50,60,70,90,120,130,140,
    145,150,155,170,175,180,185,190,195,200,205,210,215,220,225,230,235,240,245,250,260,270,280,290,300,310,
    320,330,340,350,410,
     5,10,30,50,100,200]
merger=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,
        0,0,0,0,0,0]
merger_sim=['fg1_m_13','fg1_m_13','fg1_m_13',
            'fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13',
           'fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13',
           'fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13','fg1_m_13',
            'iso_m0.333_fg0.1','iso_m0.333_fg0.1','iso_m0.333_fg0.1','iso_m0.333_fg0.1','iso_m0.333_fg0.1','iso_m0.333_fg0.1']



img_list=[ 'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_005.fits',
         'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_040.fits',
         'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_060.fits',
         'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_100.fits',
         'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_200.fits',
         'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_300.fits',
         'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_400.fits',
          'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_410.fits',
          'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_420.fits',
         'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_600.fits',
          'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_640.fits',
         'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_700.fits',
         'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_800.fits',
          'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_840.fits',
         'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_900.fits',
           'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_940.fits',
           'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_980.fits',
         'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_1000.fits',
           'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_1020.fits',
          'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_1040.fits',
           'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_1060.fits',
           'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_1080.fits',
         'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_1100.fits',
         'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_1140.fits',
           'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_1145.fits',
           'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_1150.fits',
           'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_1155.fits',
           'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_1160.fits',
           'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_1165.fits',
           'q0.1_fg0.3_BT0.2_allrx10_sunruns/hires_kin_late/broadband_1168.fits']
myr=[5,40,60,100,200,300,400,410,420,
     600,640,700,800,840,900,940,980,1000,1020,1040,1060,1080,1100,1140,
     1145,1150,1155,1160,1165,1168]
merger_sim=['fg3_m_10','fg3_m_10','fg3_m_10','fg3_m_10','fg3_m_10','fg3_m_10',
        'fg3_m_10','fg3_m_10','fg3_m_10','fg3_m_10','fg3_m_10','fg3_m_10',
        'fg3_m_10','fg3_m_10','fg3_m_10','fg3_m_10','fg3_m_10','fg3_m_10',
        'fg3_m_10','fg3_m_10','fg3_m_10','fg3_m_10','fg3_m_10','fg3_m_10',
            'fg3_m_10','fg3_m_10','fg3_m_10','fg3_m_10','fg3_m_10','fg3_m_10']
merger=[1,1,1,1,1,1,1,1,1,1,1,1,1,
       1,1,1,1,1,1,1,1,1,1,1,
        1,1,1,1,1,1]



img_list=['q0.2_fg0.3_BT0.2_allrx10_sunruns/hires_kin_early_cen1/broadband_300.fits']
myr = [300]
merger = [1]
merger_sim=['q0.2_fg0.3_BT0.2']


img_list=['q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_170.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_180.fits',
    'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_185.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_190.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_195.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_205.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_210.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_220.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_225.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_230.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_240.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_250.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_260.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_265.fits',
          'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_275.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_285.fits',
          'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_295.fits',
            'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_305.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_311.fits',
        'q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_315.fits','q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_320.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_005.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_010.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_020.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_030.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_040.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_060.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_080.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_100.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_120.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_140.fits',
         'q0.5_fg0.3_allrx10_sunruns/hires_kin_early_cen1/broadband_160.fits']
myr=[170,180,185, 190, 195, 205, 210, 220, 225, 230, 240, 250, 260,265,275,285, 295,305,311, 315,320,
     5,10,20,30,40,60,80,100,120,140,160]
merger=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,
       1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
merger_sim=['fg3_m12','fg3_m12','fg3_m12','fg3_m12',
            'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
            'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
            'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12',
           'fg3_m12','fg3_m12','fg3_m12','fg3_m12']




img_list=['isolated_galaxies/m0.5_fg0.3/broadband_005.fits','isolated_galaxies/m0.5_fg0.3/broadband_010.fits','isolated_galaxies/m0.5_fg0.3/broadband_030.fits',
          'isolated_galaxies/m0.5_fg0.3/broadband_050.fits','isolated_galaxies/m0.5_fg0.3/broadband_200.fits',
         'isolated_galaxies/m0.333_fg0.1/broadband_005.fits','isolated_galaxies/m0.333_fg0.1/broadband_010.fits','isolated_galaxies/m0.333_fg0.1/broadband_030.fits',
          'isolated_galaxies/m0.333_fg0.1/broadband_050.fits','isolated_galaxies/m0.333_fg0.1/broadband_100.fits','isolated_galaxies/m0.333_fg0.1/broadband_200.fits',
         'isolated_galaxies/m1_fg0.1/broadband_005.fits','isolated_galaxies/m1_fg0.1/broadband_010.fits','isolated_galaxies/m1_fg0.1/broadband_020.fits',
          'isolated_galaxies/m1_fg0.1/broadband_030.fits','isolated_galaxies/m1_fg0.1/broadband_040.fits','isolated_galaxies/m1_fg0.1/broadband_050.fits',
          'isolated_galaxies/m1_fg0.1/broadband_060.fits','isolated_galaxies/m1_fg0.1/broadband_100.fits','isolated_galaxies/m1_fg0.1/broadband_200.fits',
         'isolated_galaxies/m1_fg0.3/broadband_005.fits','isolated_galaxies/m1_fg0.3/broadband_010.fits','isolated_galaxies/m1_fg0.3/broadband_020.fits',
          'isolated_galaxies/m1_fg0.3/broadband_030.fits','isolated_galaxies/m1_fg0.3/broadband_040.fits','isolated_galaxies/m1_fg0.3/broadband_050.fits',
          'isolated_galaxies/m1_fg0.3/broadband_060.fits','isolated_galaxies/m1_fg0.3/broadband_100.fits','isolated_galaxies/m1_fg0.3/broadband_200.fits',
         'isolated_galaxies/m1_fg0.3_BT0.2/broadband_005.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_030.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_060.fits',
          'isolated_galaxies/m1_fg0.3_BT0.2/broadband_090.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_100.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_120.fits',
          'isolated_galaxies/m1_fg0.3_BT0.2/broadband_150.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_180.fits','isolated_galaxies/m1_fg0.3_BT0.2/broadband_200.fits']
myr=[5,10,30,50,200,
    5,10,30,50,100,200,
    5,10,20,30,40,50,60,100,200,
    5,10,20,30,40,50,60,100,200,
    5,30,60,90,100,120,150,180,200]
merger_sim=['iso_m0.5_fg0.3','iso_m0.5_fg0.3','iso_m0.5_fg0.3','iso_m0.5_fg0.3','iso_m0.5_fg0.3',
           'iso_m0.333_fg0.1','iso_m0.333_fg0.1','iso_m0.333_fg0.1','iso_m0.333_fg0.1','iso_m0.333_fg0.1','iso_m0.333_fg0.1',
           'iso_m1_fg0.1','iso_m1_fg0.1','iso_m1_fg0.1','iso_m1_fg0.1','iso_m1_fg0.1','iso_m1_fg0.1','iso_m1_fg0.1','iso_m1_fg0.1','iso_m1_fg0.1',
           'iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3',
           'iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2','iso_m1_fg0.3_BT0.2']
merger=[0,0,0,0,0,0,
       0,0,0,0,0,0,0,
       0,0,0,0,0,0,0,0,0,0,
       0,0,0,0,0,0,0,0,0,0,
       0,0,0,0,0]


img_list=['isolated_galaxies/m0.5_fg0.3/broadband_005.fits','isolated_galaxies/m0.5_fg0.3/broadband_010.fits','isolated_galaxies/m0.5_fg0.3/broadband_030.fits',
          'isolated_galaxies/m0.5_fg0.3/broadband_050.fits','isolated_galaxies/m0.5_fg0.3/broadband_200.fits',
         'isolated_galaxies/m1_fg0.3/broadband_005.fits','isolated_galaxies/m1_fg0.3/broadband_010.fits','isolated_galaxies/m1_fg0.3/broadband_020.fits',
          'isolated_galaxies/m1_fg0.3/broadband_030.fits','isolated_galaxies/m1_fg0.3/broadband_040.fits','isolated_galaxies/m1_fg0.3/broadband_050.fits',
          'isolated_galaxies/m1_fg0.3/broadband_060.fits','isolated_galaxies/m1_fg0.3/broadband_100.fits','isolated_galaxies/m1_fg0.3/broadband_200.fits']
myr=[5,10,30,50,200,
 5,10,20,30,40,50,60,100,200]
merger_sim=['iso_m0.5_fg0.3','iso_m0.5_fg0.3','iso_m0.5_fg0.3','iso_m0.5_fg0.3','iso_m0.5_fg0.3',
           'iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3','iso_m1_fg0.3']
merger=[0,0,0,0,0,0,
 0,0,0,0,0,0,0,0]


img_list=['q0.5_fg0.3_allrx10_sunruns/hires_kin/broadband_210.fits']
myr=[210]
merger=[1]
merger_sim=['fg3_m12']



import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

'''
Choose a redshift
'''
z=0.03

if len(img_list) != len(myr) or len(img_list) != len(merger) or len(img_list) != len(merger_sim):
    print(len(img_list), len(myr),len(merger), len(merger_sim) )
    stop
viewpts=[0,1,2,3,4,5,6]#,1,2,3,4,5,6]

file2=open('LDA_fg3_m13_SDSS_match_20_fake.txt','w')


counter=0



plott='no' #option for verbose outputs of some of the codes
os.chdir(os.path.expanduser('/Users/beckynevin/Documents/Backup_My_Book/My_Passport_backup/LAURA_Sims/')) # path to the simulated images

#empty arrays for outputs, also I made some for # bulges as well as myr_here and viewpts_here
#so I can flatten those later to figure out which galaxies are classified as mergers and plot their
#images together down the line.
mag_list=[]
S_N_list=[]
mag_list_after=[]
    
for k in range(len(viewpts)):
    viewpt=viewpts[k]
    for i in range(len(img_list)):
        #i=i+10
        
        print(myr[i], viewpt)
        os.chdir(os.path.expanduser('/Users/beckynevin/Documents/Backup_My_Book/My_Passport_backup/LAURA_Sims/')) # path to the simulated images

        
        plt.clf()
        '''Produce camera will use the CAMERA- extensions to figure out the pixelscale and make
        some pretty images, the output is surface brightness, use a[1][2] to get r-band,
        a[0] is the pixelscale ("/pix)'''
        im=pyfits.open(img_list[i])
        
        camera_data=im['CAMERA'+str(viewpt)+'-BROADBAND'].data
        
        pixelscale =  im['CAMERA'+str(viewpt)+'-BROADBAND'].header['CD1_1']
        
        
        #print(im['SFRHIST'].header['SNAPTIME']/10**9)
        myr_actual=im['SFRHIST'].header['SNAPTIME']/10**9
        
        
        a=produce_camera(img_list[i],viewpt)
        #return pixelscale, camera_data --> first dimension of this is the camera
        
        
        '''The next few functions essentially do what Source Extractor (Sextractor) does.
        However, I wanted to understand exactly how the low, high, and sharpening filters
        work and this seems to work well for my purposes, which are to obtain a good guess
        at a Galfit input. Overall, they determine which is the brighter source, center the
        camera on this source, and cutout the image.'''
        
        '''I later use Source Extractor to double check this, and sometimes the off-center
        aperture is brighter, so I re-center and use determine_coords again.'''

        '''Determine coords will use a combination of a low-pass and a high-pass and a 
        sharpening filter to identify the coordinates of the two brightest points in the 
        r-band image.'''
        band=65
        try:
            fill_img=a[1][band]
        except IndexError:
            band=2
        b=determine_coords(a[1][band])
        #this determines the locations of the galaxies
        
        if b[0]==0:
            #if the first index from determine_coords is zero, the galaxy is out of the image
            #and we can continue and skip this particular image (sometimes other viewpoints of
            #the same snapshot are in the frame so don't get rid of an entire snapshot)
            continue
        
        
        
 
        
        
        
        low_pass=b[4]
        num_sol=b[5]

        '''Now, fit a couple 2D gaussians if there are 2 brightest pixels, otherwise
        fit only one 2D gaussian. The output of fit_2_gaussians will be the positions of these
        maxima'''

        if num_sol==1:
            #this is if there's only really one solution because the bulges are too close together
            #fit a 2D gaussian to center the previous guess of peak pixels using the entire surface
            #brightness profile
            c=fit_2_gaussian(b[1],shape(a[1][band])[0]-b[0],b[1],shape(a[1][band])[0]-b[0],low_pass)
            if c[8]=='no':
                 c=fit_2_gaussian(b[2],shape(a[1][band])[0]-b[0],b[3],shape(a[1][band])[0]-b[1],low_pass)

        else:
            c=fit_2_gaussian(b[2],shape(a[1][band])[0]-b[0],b[3],shape(a[1][band])[0]-b[1],low_pass)

        
        
        if c[4] > c[5]:
            '''this means point 1 is brighter'''
            in_x = c[1]
            in_y = c[0]
            in_2_x = c[3]
            in_2_y = c[2]
            

        if c[5] > c[4]:
            '''point 2 is the brighter source'''
            in_x = c[3]
            in_y = c[2]
            in_2_x = c[1]
            in_2_y = c[0]

            
        
        
        '''Now place a aperture over each center and figure out which is brighter overall'''
        c_final=determine_brighter(a[1][band],  in_y,shape(a[1][band])[0]-in_x,  in_2_y, shape(a[1][band])[0]-in_2_x, a[0], z)
       

        if c_final[0] > c_final[1]:
            #Clip the image in the SDSS imaging size around 
            d=clip_image(a[1][band], a[0], z, int(shape(a[1][band])[0]-in_x), int(in_y))
            #this means the first aperture (center) is indeed brighter
        else:
            d=clip_image(a[1][band], a[0], z, int(shape(a[1][band])[0]-in_2_x), int(in_2_y))

   
        if d[3]=='no' or d[6]==0:
            #this means the cutout is outside the image --> bad
            print('Outside of Box, continuing')
            continue
        
        
        '''Now we need to clip this so that its the size of an SDSS image 
        (50" by 50" in diameter)'''
        '''You also need to use the redshift to get an accurate cutout, here its 0.03, the
        mean redshift of MaNGA'''

        
        

        #def convolve_rebin_image(number, z,pixscale, view, counts):#all of these are in nanomags

    

        '''d output is surface brightness in W/m/m^2/sr but cut'''
        
        '''Next, convert to this weird nanomag unit that SDSS tends to use:
        The standard source for each SDSS band is close to but not exactly 
        the AB source (3631 Jy), meaning that a nanomaggy is approximately 3.631×10-6 Jy,
        which is the conversion we use here.
        Note that nanomaggies are still a unit of surface brightness so we can use nanomaggies
        later in the code to compute the nonparametric quantities Gini, M20, C, A, and S.'''
        '''Nanomags also does a couple of important image processing steps like introduce a sky
        background that is typical for SDSS imaging as well a constructs a Poisson noise map.'''
        e=nanomags(z, a[0], d[0],viewpt, myr[i])
       
        

        

    
        texp=1
        prep_out = convolve_rebin_image(myr[i],z, a[0], viewpt, e[4], texp, d[6])
        prep = prep_out[0]
        prep_sigma = prep_out[1]
        prep_cut = prep_out[2]
        prep_sigma_cut = prep_out[3]


        
        
 
        
        
 
        
        
        #return n, sky resids added (nanomags), poisson noise, resids in nanomags, sky resids added (counts), poisson noise, resids in counts
        #return nanomaggy, degraded_image, sigma_nanomags, sky_resids_mine, degraded_image_counts, sigma_counts, sky_resids_mine_counts

        #e[0] is nanomaggy
        #e[1] is degraded nano
        #e[2] is poisson nano
        #e[3] is sky resids nano
        #e[4] is degraded counts --> counts with background resids added (this is most important!!)
        #e[5] is poisson counts
        #e[6] is sky resids counts
        
        '''However, the input to galfit is in counts, so since these are simulations, we can
        set the exposure time equal to anything we want, I chose 1 because I think it really
        crashes Galfit if you have too many counts and texp comes in later as a multiplicative 
        factor.
        Convolve image also does the final step of image processing'''
        
        #prep=convolve_image(myr[i],e[0],e[1],e[2],z,a[0],viewpt, e[3], e[4], e[5], e[6], texp)
        '''Prep saves the galfit input files and outputs a convolved file in counts'''
        
        #continue
        '''Running sextractor before galfit to determine good guess input parameters'''
        
        outfile = 'q0.5_fg0.3_allrx10/fg3_m12_rebin/pet_radius_'+str(myr[i])+'_'+str(viewpt)+'.fits'
        
        hdu = fits.PrimaryHDU(prep)#was d[0] (the unconvolved version)
        hdu_number = 0
        hdu.writeto(outfile, overwrite=True)
        hdr=fits.getheader(outfile, hdu_number)

        hdr['EXPTIME'] = 1
        hdr['EXPTIME']

        hdu.writeto(outfile, overwrite=True)

        #continue

        outfile = 'q0.5_fg0.3_allrx10/pet_radius_'+str(myr[i])+'_'+str(viewpt)+'.fits'
        
        hdu = fits.PrimaryHDU(prep)#was d[0] (the unconvolved version)
        hdu_number = 0
        hdu.writeto(outfile, overwrite=True)
        hdr=fits.getheader(outfile, hdu_number)

        hdr['EXPTIME'] = 1
        hdr['EXPTIME']

        hdu.writeto(outfile, overwrite=True)

        
        
        
        write_sex_default(myr[i], viewpt)
        run_sex(myr[i],viewpt)
        sex_out=sex_params_galfit(myr[i],viewpt)
        if sex_out[0]==0:
            continue
        plot_sex(myr[i],viewpt, sex_out[3], sex_out[4])
        
 
        
        
        '''Step 1: check if the center really is the center from source extractor'''
        if np.sqrt((sex_out[3]-np.shape(prep)[0]/2)**2+(sex_out[4]-np.shape(prep)[0]/2)**2) > np.sqrt(200):
            '''Return and recenter the cutout on the second center since its brigher
            according to sextractor'''
            #print('Source Extractor got the Wrong Source')
            os.chdir(os.path.expanduser('/Users/beckynevin/Documents/Backup_My_Book/My_Passport_backup/LAURA_Sims/')) # path to the simulated images
            try:
                d=clip_image(a[1][band], a[0], z, int(sex_out[11]+((shape(a[1][band])[0]-in_2_x)-d[2])), int(sex_out[12]+(in_2_y-d[2])))
            except IndexError:
                continue
            print(d[3])
            if d[3]=='no' or d[6]==0:
                #this means the cutout is outside the image --> bad
                print('Outside of Box, continuing')
                continue

            #See above
            e=nanomags(z, a[0], d[0],viewpt, myr[i])#was a[1]

            
            texp=1
            prep_out = convolve_rebin_image(myr[i],z, a[0], viewpt, e[4], texp, d[6])
            prep = prep_out[0]
            prep_sigma = prep_out[1]
            prep_cut = prep_out[2]
            prep_sigma_cut = prep_out[3]
            '''Prep saves the galfit input files'''


            '''Running sextractor before galfit'''

            outfile = 'q0.5_fg0.3_allrx10/pet_radius_'+str(myr[i])+'_'+str(viewpt)+'.fits'
            hdu = fits.PrimaryHDU(prep)#was d[0] (the unconvolved version)
            hdu_number = 0
            hdu.writeto(outfile, overwrite=True)
            hdr=fits.getheader(outfile, hdu_number)

            hdr['EXPTIME'] = 1
            hdr['EXPTIME']

            hdu.writeto(outfile, overwrite=True)

            write_sex_default(myr[i], viewpt)
            run_sex(myr[i],viewpt)
            sex_out=sex_params_galfit(myr[i],viewpt)
            if sex_out[0]==0:
                continue
        
        
        
        #return sex_pet_r, minor, flux, x_max, y_max, mag_max, n_bulges, eff_radius_1, B_A_1, PA_1, x_max_2, y_max_2, mag_max_2, eff_radius_2, B_A_2, PA_2
        #print(img_list[i], d[4], d[5])
        '''So if theres only one bulge we will get a different set of output parameters'''
        
        length_gal=shape(prep)[0]#shape(d[0])[0]
        
        

        '''now put together all of the necessary Galfit inputs - it is a little finnicky and needs to know
        magnitude guesses, which is annoying because everything is in counts and often there's no exposure time
        provided, but you can guess the magnitude as below by setting the mag_zpt and then calculating the flux ratio
        given the source extractor background and main source counts'''

        
     

        '''This is scaled up using the background flux and the flux from the petrosian output'''
        
        '''Now use the output of source extractor to also determine the petrosian radius,
        the semimajor and semiminor axes, the flux guess, the bg level, the PA, and the axis ratio'''
        mag_zpt=26.563
        #try setting mag_zpt as the background magnitude and scale up from there with the flux
        num_bulges=sex_out[6]
        
        '''This is scaled up using the background flux and the flux from the petrosian output'''
        mag_guess=(-2.5*np.log10((sex_out[5])/sex_out[10])+mag_zpt)
        
        
        if num_bulges==2:
            #def write_galfit_feedme(view,number,xcen,ycen,xcen2,ycen2, mag, mag_zpt, num_bulges, length_gal, r_1, r_2, mag_2, B_A_1, B_A_2, PA_1, PA_2):
            #return sex_pet_r, minor, flux, x_max, y_max, mag_max, n_bulges, eff_radius_1, B_A_1, PA_1,back_1, x_max_2, y_max_2, mag_max_2, eff_radius_2, B_A_2, PA_2, back_2
            mag_guess=(-2.5*np.log10((sex_out[5])/sex_out[10])+mag_zpt)
                
            mag_guess_2=(-2.5*np.log10((sex_out[13])/sex_out[10])+mag_zpt)
            
            f=write_galfit_feedme(viewpt,myr[i], sex_out[3], length_gal+1-sex_out[4], sex_out[11], length_gal+1-sex_out[12], mag_guess, mag_zpt, num_bulges, length_gal, sex_out[7], sex_out[14], mag_guess_2, sex_out[8], sex_out[9], sex_out[15], sex_out[16], sex_out[10], z , a[0])#was 300-in_x
        else:
            f=write_galfit_feedme(viewpt,myr[i], sex_out[3], length_gal+1-sex_out[4], 0,0, mag_guess, mag_zpt, num_bulges, length_gal, sex_out[7],0,0, sex_out[8], sex_out[9], 0, 0, sex_out[10], z, a[0])#was 300-in_x
            
        '''Runs galfit inline and creates an output file'''
        g=run_galfit(viewpt,myr[i])


        continue
        output='GALFIT_folder/out_'+str(viewpt)+'_'+str(myr[i])+'.fits'
        try:
            #This is the output result fit from Galfit:
            out=pyfits.open(output)
        except FileNotFoundError:
            #sometimes Galfit fails and you need to move on with your life.
            #It should really work for the majority of cases though.
            print('Galfit failed', viewpt,myr[i])
            continue
        try:
            
            h=galfit_params(viewpt,myr[i],num_bulges,a[0])

            

            PA1=h[2]
            PA2=h[3]
            posx_1=h[4]
            posy_1=h[5]
            posx_2=h[6]
            posy_2=h[7]
            AR_1=h[8]
            gal_out=h[9]
            ser=h[10]

        except ValueError:
            continue
        
        
        '''Now use the output of source extractor to also determine the petrosian radius'''
        ##return sex_pet_r, minor, flux, x_max, y_max, mag_max, n_bulges, eff_radius_1, B_A_1, PA_1, x_max_2, y_max_2, mag_max_2, eff_radius_2, B_A_2, PA_2
       
        r_sex=sex_out[0]
        b_sex=sex_out[1]
        flux_sex=sex_out[2]
        bg_level=sex_out[10]

        '''can calculate the pet magnitude this way to compare'''
        #print('mag from e[7]',e[7])
        #print('flux at petrosian', flux_sex)
        Area=(b_sex*r_sex*np.pi)
        mag_flux = -2.5*np.log10(((flux_sex)*0.005*3.631*10**(-6))/3631)
        mag_flux_Area = -2.5*np.log10(((flux_sex/Area)*0.005*3.631*10**(-6))/3631)
        
        
        #print('BG level sex', bg_level)
        #print('sex things', r_sex, b_sex, flux_sex)

        '''limiting_sb=input_petrosian_radius(r_sex, b_sex, flux_sex)
        threshold = np.array(limiting_sb*np.ones((np.shape(prep)[0],np.shape(prep)[1])))
        #print(threshold)
        npixels = 1  # was 1 minimum number of connected pixels
        #threshold=0.1

    

        segm = photutils.detect_sources(prep, threshold, npixels)
        
        label = np.argmax(segm.areas) + 1
        segmap = segm.data == label

        segmap_float = ndimage.uniform_filter(np.float64(segmap), size=10)

        new_threshold = (0.75*(np.max(segmap_float)-np.min(segmap_float))+np.min(segmap_float))
        segm = photutils.detect_sources(segmap_float, new_threshold, npixels, connectivity=8)

        label = np.argmax(segm.areas) + 1
        
        segmap = segm.data == label

        segmap_float = ndimage.uniform_filter(np.float64(segmap), size=10)
        segmap = segmap_float >  0.5'''

    


        '''This is vicente's method'''
        threshold = photutils.detect_threshold(prep, snr=1.5)#, background=0.0)#was 1.5
        npixels = 5  # minimum number of connected pixels was 5
        segm = photutils.detect_sources(prep, threshold, npixels)

        try:
            label = np.argmax(segm.areas) + 1
        except ValueError:
            continue
        segmap = segm.data == label
        import scipy.ndimage as ndi

        segmap_float = ndi.uniform_filter(np.float64(segmap), size=10)
        segmap = segmap_float > 0.5

        '''Vicente says to do some background estimation'''
        '''from astropy.stats import SigmaClip
        from photutils import Background2D, MedianBackground
        sigma_clip = SigmaClip(sigma=3., iters=10)
        bkg_estimator = MedianBackground()
        try:
            bkg = Background2D(prep, (126,126),edge_method='pad',
                           filter_size=(3,3),sigma_clip=sigma_clip,bkg_estimator=bkg_estimator)
        except ValueError:
            continue'''

        '''       my_bg = np.median([np.median(prep[0:5,0:5]),np.median(prep[121:126,121:126]),
                           np.median(prep[121:126,0:5]),np.median(prep[0:5,121:126])])

        print('my bg', my_bg, 'sex bg', bg_level, 'background2d', np.median(bkg.background))

 
        plt.clf()
        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        im1 = ax1.imshow(prep)
        plt.colorbar(im1)

        ax2 = fig.add_subplot(312)
        im2 = ax2.imshow(bkg.background, vmin=-10, vmax=10)
        plt.colorbar(im2)
        #im4 = ax2.imshow(prep[121:126,121:126], vmin=-10, vmax=10)
        #im5 = ax2.imshow(prep[0:5,121:126], vmin=-10, vmax=10)
        #im6 = ax2.imshow(prep[121:126,0:5], vmin=-10, vmax=10)
        
        

        ax3 = fig.add_subplot(313)
        im3 = ax3.imshow(prep-my_bg)
        plt.colorbar(im3)
        plt.savefig('background_subtract_'+str(myr[i])+'_'+str(viewpt)+'.pdf')'''

        '''plt.clf()
        fig=plt.figure()
        ax1=fig.add_subplot(211)
        im1=ax1.imshow(prep)
        plt.colorbar(im1)
    
        ax2=fig.add_subplot(212)
        im2=ax2.imshow(prep_sigma)
        plt.colorbar(im2)
        plt.savefig('diagnostic_sig_mock.pdf')
        STOP'''
        


        try:
            source_morphs = statmorph.source_morphology(prep, segmap,
                                                         weightmap=prep_sigma)#, skybox_size=25)#, skybox_size=10)
        except ValueError:
            continue
        try:
            morph=source_morphs[0]
        except IndexError:
            continue
        

        '''print('r20 = ',morph.r20)
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
        print('my sersic', ser)'''


        '''plt.clf()
        fig = make_figure(morph)
        fig.savefig('template_fg3_m1_10_'+str(myr[i])+'_'+str(viewpt)+'.png')'''
 
        if morph.flag==1: 
 
            continue
        S_N_list.append(morph.sn_per_pixel)
        mag_list.append(e[7])
        mag_list_after.append(mag_flux)

        if morph.sn_per_pixel < 2.5:
            continue
        if morph.sn_per_pixel < 4:
            plt.clf()
            fig=make_figure(morph)
            fig.savefig('S_N_4_'+str(myr[i])+'_'+str(viewpt)+'.png')
            #STOP
        

        j=morph.gini
        jj=morph.m20
        
        
        kk=morph.concentration
        l=morph.smoothness
        
        
        gg=morph.asymmetry

        
        n=morph.shape_asymmetry

        '''import numpy.ma as ma

        gal_zoom = ma.masked_where(segmap==0, prep)
        gal_zoom_bg = ma.masked_where(segmap==0, bkg.background)
        my_asy = asymmetry(gal_zoom, gal_zoom_bg)'''
        #print('morph assy', gg)
        #print('my asy', my_asy)

        #plt.clf()
        #fig=make_figure(morph)
        #plt.savefig('diagnostic_mock_seg_expanded_'+str(myr[i])+'_'+str(viewpt)+'.png')
        
        plt.clf()
        fig, ax1 = plt.subplots(figsize=(5,5), facecolor='black')
 

        im1=ax1.imshow(abs(prep), norm=matplotlib.colors.LogNorm(vmin=10**(0), vmax=10**4),cmap='afmhot')#norm=matplotlib.colors.LogNorm(vmin=10**(-1), vmax=10**2)
        ticks = [0,np.shape(prep)[0]/2, np.shape(prep)[0]]
        tick_labels = [int(round(-0.396*np.shape(prep)[0]/2,0)),0.0,int(round(0.396*np.shape(prep)[0]/2,0))]
        ax1.set_yticks(ticks)
        ax1.set_yticklabels(tick_labels, size=15)


        ax1.set_xticks(ticks)
        ax1.set_xticklabels(tick_labels, size=15)

        ax1.set_ylabel(r'Spatial Position [$^{\prime\prime}$]', size=15)
        ax1.set_xlabel(r'Spatial Position [$^{\prime\prime}$]', size=15)
        
        #ax1.axis('off')

        #cbar = fig.colorbar(im1, cax=cax)
        #cbar.ax.tick_params(labelsize=15)
        #cbar.set_label('Counts', size=15)
        
#        ax1.annotate(r't = '+str(round(myr_actual,2))+' Gyr', xy=(0.05,0.9), color='white', xycoords='axes fraction', size=20)
        plt.tight_layout()
        ax1.axis('off')
        plt.savefig('../MaNGA_Papers/Paper_I/SDSS_image_animate_gini_'+str(round(myr_actual,2))+'_'+str(viewpt)+'_'+str(merger_sim[i])+'.png',bbox_inches='tight')
        

        plt.clf()
        fig, ax1 = plt.subplots(figsize=(5,5))
 

        im1=ax1.imshow(abs(prep), norm=matplotlib.colors.LogNorm(vmin=10**(0), vmax=10**4),cmap='afmhot')#norm=matplotlib.colors.LogNorm(vmin=10**(-1), vmax=10**2)
        ticks = [0,np.shape(prep)[0]/2, np.shape(prep)[0]]
        tick_labels = [int(round(-0.396*np.shape(prep)[0]/2,0)),0.0,int(round(0.396*np.shape(prep)[0]/2,0))]
        ax1.set_yticks(ticks)
        ax1.set_yticklabels(tick_labels, size=15)


        ax1.set_xticks(ticks)
        ax1.set_xticklabels(tick_labels, size=15)

        ax1.set_ylabel(r'Spatial Position [$^{\prime\prime}$]', size=15)
        ax1.set_xlabel(r'Spatial Position [$^{\prime\prime}$]', size=15)
        
        #ax1.axis('off')

        #cbar = fig.colorbar(im1, cax=cax)
        #cbar.ax.tick_params(labelsize=15)
        #cbar.set_label('Counts', size=15)
        
        ax1.annotate(r't = '+str(round(myr_actual,2))+' Gyr', xy=(0.05,0.9), color='white', xycoords='axes fraction', size=20)
        plt.tight_layout()
        ax1.axis('off')
        if myr[i] < 10:
            plt.savefig('../MaNGA_Papers/Paper_I/SDSS_image_animate_00'+str(myr[i])+'_'+str(viewpt)+'_'+str(merger_sim[i])+'.png',bbox_inches='tight')
        else:
            if myr[i]< 100:
                plt.savefig('../MaNGA_Papers/Paper_I/SDSS_image_animate_0'+str(myr[i])+'_'+str(viewpt)+'_'+str(merger_sim[i])+'.png',bbox_inches='tight')
            else:
                plt.savefig('../MaNGA_Papers/Paper_I/SDSS_image_animate_'+str(myr[i])+'_'+str(viewpt)+'_'+str(merger_sim[i])+'.png',bbox_inches='tight')

        
       
        
        plt.clf()
        #fig, (ax1, cax) = plt.subplots(ncols=2, figsize=(5,5),
#                                  gridspec_kw={"width_ratios":[1,0.1]})
#        fig.subplots_adjust(wspace=0.3)
        fig=plt.figure()
        ax1=fig.add_subplot(111)
        import heapq
        array_d=d[0].flatten()
        max_img=(heapq.nlargest(5, array_d)[2]/4)
        mask_log_image=d[0]#ma.masked_where(d[0]<0, d[0])
        #mask_log_image=mask_log_image.filled(0.05)
        
        #im1=ax1.imshow((mask_log_image),vmax=20,cmap='afmhot')#inferno is very pink/orange
        '''Need to cut it down to the OG size'''

        #factor = 25/0.396
        
        #im1=ax1.imshow(abs(prep_cut[int(np.shape(prep_cut)[0]/2-factor):int(np.shape(prep_cut)[0]/2+factor),int(np.shape(prep_cut)[0]/2-factor):int(np.shape(prep_cut)[0]/2+factor)]), norm=matplotlib.colors.LogNorm(vmin=10**(0), vmax=10**3),cmap='afmhot')#norm=matplotlib.colors.LogNorm(vmin=10**(-1), vmax=10**2)
        #ticks = [0,factor, 2*factor]
        #tick_labels = [-25.0,0.0,25.0]

        im1=ax1.imshow(abs(prep), norm=matplotlib.colors.LogNorm(vmin=10**(0), vmax=10**4),cmap='afmhot')#norm=matplotlib.colors.LogNorm(vmin=10**(-1), vmax=10**2)
        ticks = [0,np.shape(prep)[0]/2, np.shape(prep)[0]]
        tick_labels = [int(round(-0.396*np.shape(prep)[0]/2,0)),0.0,int(round(0.396*np.shape(prep)[0]/2,0))]
        ax1.set_yticks(ticks)
        ax1.set_yticklabels(tick_labels, size=15)


        ax1.set_xticks(ticks)
        ax1.set_xticklabels(tick_labels, size=15)

        ax1.set_ylabel(r'Spatial Position [$^{\prime\prime}$]', size=15)
        ax1.set_xlabel(r'Spatial Position [$^{\prime\prime}$]', size=15)
        
        ax1.axis('off')

        #cbar = fig.colorbar(im1, cax=cax)
        #cbar.ax.tick_params(labelsize=15)
        #cbar.set_label('Counts', size=15)
        
        #ax1.annotate(r't = '+str(round(myr_actual,2))+' Gyr', xy=(0.4,0.9), color='white', xycoords='axes fraction', size=15)
        plt.tight_layout()
        #plt.title(str(mag_flux))
        plt.savefig('../MaNGA_Papers/Paper_I/SDSS_image_plot_contrast_rebin_'+str(myr[i])+'_'+str(viewpt)+'_'+str(merger_sim[i])+'.png',bbox_inches='tight')
        continue
           
        
 
        
        if counter==0:
            file2.write('Counter'+'\t'+'Image'+'\t'+'Merger'+'\t'+
                'Myr'+'\t'+'Viewpoint'+'\t'+'# Bulges'+'\t'+'Sep'+'\t'+'Flux Ratio'+'\t'+'Gini'+'\t'+'M20'+'\t'+'C'+'\t'+'A'+'\t'+'S'+
                        '\t'+'Sersic n'+'\t'+'A_s'+'\n')
            #Counter	Image	Merger	Myr	Viewpoint	# Bulges	Sep	Flux Ratio	Gini	M20	C	A	S	Sersic n	A_s

        
        file2.write(str(counter)+'\t'+str(img_list[i])+'\t'+str(merger[i])+'\t'+str(round(myr_actual,2))+'\t'+str(viewpt)+'\t'+str(num_bulges)+'\t'+str(h[0])+'\t'+str(h[1])+'\t'+str(j)+'\t'+str(jj)+'\t'+str(kk)+'\t'+str(gg)+'\t'+str(l)+'\t'+str(ser)+'\t'+str(n)+'\n')#was str(np.shape(vel_dist)[1]-1-j)
        #print('these are your variables', str(counter)+'\t'+str(img_list[i])+'\t'+str(merger[i])+'\t'+str(round(myr_actual,2))+'\t'+str(viewpt)+'\t'+str(num_bulges)+'\t'+str(h[0])+'\t'+str(h[1])+'\t'+str(j)+'\t'+str(jj)+'\t'+str(kk)+'\t'+str(gg)+'\t'+str(l)+'\t'+str(ser)+'\t'+str(n)+'\n')
        counter +=1
        
 
        os.system('rm q0.5_fg0.3_allrx10/pet_radius_'+str(myr[i])+'_'+str(viewpt)+'.fits')
        os.system('rm GALFIT_folder/out_'+str(viewpt)+'_'+str(myr[i])+'.fits')
        os.system('rm GALFIT_folder/out_sigma_convolved_'+str(viewpt)+'_'+str(myr[i])+'.fits')
        #os.system('rm GALFIT_folder/out_convolved_'+str(viewpt)+'_'+str(myr[i])+'.fits')
        os.system('rm GALFIT_folder/galfit.feedme_'+str(viewpt)+'_'+str(myr[i]))
        #STOP
print(mag_list)
print(np.mean(mag_list), np.std(mag_list))


print(mag_list_after)
print(np.mean(mag_list_after), np.std(mag_list_after))

print(S_N_list)

plt.clf()
plt.hist(S_N_list, bins=20)
plt.axvline(x=2.5)
plt.axvline(x=4)
plt.xlim([0,10])
plt.title('S/N list')
plt.xlabel('S_N per pixel')
plt.tight_layout()
plt.savefig('S_N_hist_fg1_m1_10_deltam_2.png')

plt.clf()
plt.scatter(S_N_list, mag_list_after)
plt.xlabel('S/N')
plt.ylabel('App mag')
plt.tight_layout()
plt.savefig('S_N_mag_scatter_fg1_m1_10_deltam_2.png')
        
        
file2.close()

print('finished')

