# Takes all of the primary, secondary, color_enhanced sample galaxies from DR15 (public) and puts them in a table
# so that I can then go and grab their images from the main SDSS database

from marvin import config
from marvin.utils.general.general import get_drpall_table
config.access = 'public'#diff from collab
config.setRelease('DR15')
data = get_drpall_table()

import numpy as np
primary        = data['mngtarg1'] & 2**10
secondary      = data['mngtarg1'] & 2**11
color_enhanced = data['mngtarg1'] & 2**12

main_sample = np.logical_or.reduce((primary, secondary, color_enhanced))

plateifus = data['plateifu'][main_sample]
print(data.columns)
cols = data['nsa_camcol'][main_sample]
print(data['nsa_iauname'][main_sample])


# Make a new table with a lot of this info
file=open('/Users/beckynevin/CfA_Code/Kinematics_and_Imaging_Merger_Identification/Identify_MaNGA_SDSS_mergers_imaging/table_manga_gals.txt','w')
# Some of the galaxies fail and I have not had to time to look into why
# error is -
# ['failed to retrieve data using input parameters.']
# ['get_nsa_data: cannot find a valid DB connection.']


for j in range(len(plateifus)):
    if j==0:
        file.write('PLATEIFU'+'\t'+'run'+'\t'+'field'+'\t'+
            'camcol'+'\t'+'RA'+'\t'+'DEC'+'\t'+'redshift'+'\n')
    if data['nsa_iauname'][main_sample][j]=='-9999':
        continue
    plateifu = data['plateifu'][main_sample][j]
    ra =str(data['nsa_iauname'][main_sample][j])[1:3]+':'+str(data['nsa_iauname'][main_sample][j])[3:5]+':'+str(data['nsa_iauname'][main_sample][j])[5:10]
    
    dec =str(data['nsa_iauname'][main_sample][j])[10:13]+':'+str(data['nsa_iauname'][main_sample][j])[13:15]+':'+str(data['nsa_iauname'][main_sample][j])[15:]
    
    
    
    file.write(str(plateifu)+'\t'+
        str(data['nsa_run'][main_sample][j])+'\t'+
        str(data['nsa_camcol'][main_sample][j])+'\t'+
        str(data['nsa_field'][main_sample][j])+'\t'+
        str(ra)+'\t'+
        str(dec)+'\t'+
        str(data['nsa_z'][main_sample][j])+'\n')
            
    print(j)
file.close()


