# This code finds a way to extract and write a .txt file of MaNGA ids in order to get r-band images.

# The following is the correct formatting that we need the lists in
# SDSS superclean merger Galaxy Zoo
#sdss_list=[587730773351923978, 587727177912680538, 587727223561257225, 587731187277955213, 587730774962864138, 587727178986750088, 588015510343712908, 587730775499997384, 587727180060688562, 587727225690783827, 587727226227720294, 587731186204868616, 588015510344040607, 588015508733624406, 587727225691242683, 587730772816167051, 588015510344302646, 588015509807497271, 587727178450927744, 588290881638760664, 587727178988191864]
#ra_list=['00:00:20.24', '00:00:37.17', '00:02:27.29', '00:02:49.07', '00:03:08.23', '00:03:57.92', '00:05:27.39', '00:05:54.09', '00:05:58.00', '00:06:07.51', '00:06:24.76', '00:08:27.81', '00:08:32.88', '00:10:11.15', '00:10:33.29', '00:10:55.50', '00:11:08.83', '00:11:43.68', '00:13:30.75', '00:14:31.15', '00:16:54.99']
#dec_list=['+14:11:09.9', '-11:02:07.6', '+16:04:42.5', '+00:45:04.8', '+15:33:48.8', '-10:20:44.3', '+00:50:48.1', '+15:53:42.8', '-09:21:17.8', '-10:30:32.7', '-10:01:31.2', '-00:00:17.7', '+01:02:20.1', '-00:14:30.8', '-10:36:10.9', '+13:52:49.7', '+00:50:43.7', '+00:31:22.5', '-10:43:17.6', '+15:49:02.2', '-10:23:44.0']
import astropy.io.fits as pyfits
import marvin
from marvin import config, marvindb
from marvin.tools import Cube
#%matplotlib inline
# Make sure you have collaborator access - if not go set it up following these instructinos:
# https://sdss-marvin.readthedocs.io/en/stable/installation.html
config.access = 'collab'
# Choose the data release you would like to use (could also use MPL)
#config.setRelease('MPL-6')
print(config.access)

# I had to re-log in when I first ran this code:
config.login(refresh=True)
print(config.token)

dapall = pyfits.open('/Users/beckynevin/Clone_Docs_old_mac/Backup_My_Book/My_Passport_backup/Kinematic_ML/dapall-v2_5_3-2.3.0.fits')

print(dapall[1].data[0])
# So the MaNGA data tables have a MaNGA ID and an RA and dec, I am wondering if we could back out the SDSS id from the GalaxyZoo file.



plateifu = dapall[1].data['PLATEIFU'][0]#8485-1901'
print('plateifu', plateifu)
cube = Cube(plateifu=plateifu)
print(cube)
print(list(cube.nsa.keys()))
print(cube.nsa.ra, cube.nsa.dec, cube.nsa.iauname, cube.nsa.isdss)
print(cube.nsa.run, cube.nsa.camcol, cube.nsa.field, cube.nsa.nsaid)

# Make a new table with a lot of this info
file=open('/Users/beckynevin/CfA_Code/Kinematics_and_Imaging_Merger_Identification/Identify_MaNGA_SDSS_mergers_imaging/table_manga_gals.txt','w')
for j in range(len(dapall[1].data['PLATEIFU'])):
    if j==0:
        file.write('PLATEIFU'+'\t'+'run'+'\t'+'field'+'\t'+
            'camcol'+'\t'+'RA'+'\t'+'DEC'+'\n')
    plateifu = dapall[1].data['PLATEIFU'][j]#8485-1901'
    print('plateifu', plateifu)
    cube = Cube(plateifu=plateifu)
    print(cube.nsa.iauname)
    print(str(cube.nsa.iauname)[1:10])
    
    ra =str(cube.nsa.iauname)[1:3]+':'+str(cube.nsa.iauname)[3:5]+':'+str(cube.nsa.iauname)[5:10]
    
    dec =str(cube.nsa.iauname)[10:13]+':'+str(cube.nsa.iauname)[13:15]+':'+str(cube.nsa.iauname)[15:]
    
    
    try:
        file.write(str(plateifu)+'\t'+
            str(cube.nsa.run)+'\t'+
            str(cube.nsa.camcol)+'\t'+
            str(cube.nsa.field)+'\t'+
            str(ra)+'\t'+
            str(dec)+'\n')
    except KeyError:
        print(cube.nsa)
        STOP
    if j > 100:
        file.close()
        STOP
file.close()
