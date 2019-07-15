'''
This snippet digs through the GalaxyZoo catalog looking for various types of galaxies.
I've set a threshold of 0.95 to identify mergers, ellipticals, spirals.
It prints their objectid, RA, and dec, which are the inputs to GalaxySmelter_real_SDSS.py
'''

import pyfits
all_zoo=pyfits.open('GalaxyZoo1_DR_table2.fits')

'''These are the headers'''

print(all_zoo[1].header['TTYPE1'])
print(all_zoo[1].header['TTYPE2'])
print(all_zoo[1].header['TTYPE3'])
print(all_zoo[1].header['TTYPE4'])
print(all_zoo[1].header['TTYPE5'])
print(all_zoo[1].header['TTYPE6'])
print(all_zoo[1].header['TTYPE7'])
print(all_zoo[1].header['TTYPE8'])
print(all_zoo[1].header['TTYPE9'])
print(all_zoo[1].header['TTYPE10'])
print(all_zoo[1].header['TTYPE11'])
print(all_zoo[1].header['TTYPE12'])
print(all_zoo[1].header['TTYPE13'])
print(all_zoo[1].header['TTYPE14'])
print(all_zoo[1].header['TTYPE15'])
print(all_zoo[1].header['TTYPE16'])

'''I am looking for p_el (prob ellitpical), p_cs (prob spiral), p_mg (prob merger)'''

num_mergers=0
objid_merg=[]
ra_merg=[]
dec_merg=[]
for j in range(len(all_zoo[1].data['P_MG'])):
    if num_mergers > 300:
        break
    if all_zoo[1].data['P_MG'][j] > 0.95:# and all_zoo[1].data['P_MG'][j] < 0.5:# and all_zoo[1].data['P_MG'][j] > 0.01:
        #print(all_zoo[1].data[j]['OBJID'],all_zoo[1].data[j]['P_MG'],all_zoo[1].data[j]['P_EL'],all_zoo[1].data[j]['P_CS'], all_zoo[1].data[j]['Elliptical'],all_zoo[1].data[j]['Spiral'])
        objid_merg.append(all_zoo[1].data[j]['OBJID'])
        ra_merg.append(str(all_zoo[1].data[j]['RA']))
        dec_merg.append(str(all_zoo[1].data[j]['DEC']))
        num_mergers+=1
print(objid_merg)
print(ra_merg)
print(dec_merg)
