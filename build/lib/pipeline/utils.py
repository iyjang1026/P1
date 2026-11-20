import glob
import os, sys
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

def mkdir(path, name):
    if not os.path.exists(path + '/'+name):
        os.mkdir(path +'/'+ name)
    return str(path+'/'+name)

def file_list(path, ext_type=0):
    ext = None
    if ext_type == 0:
        ext = '.fits'
        #file = sorted(glob.glob(path + '/*.fits'))
    elif ext_type == 1:
        ext = '.fit'
        #file = sorted(glob.glob(path + '/*.fit'))
    elif ext_type == 2:
        ext = '.csv'
        #file = sorted(glob.glob(path + '/*.csv'))
    elif ext_type == 3:
        ext = '.wcs'
        #file = sorted(glob.glob(path + '/*.wcs'))
    else :
        raise ValueError('it is not supported extension')
    file = sorted(glob.glob(path + '/*'+ext))
    if len(file) == 0:
        raise NameError(f'There is no {path} directory or files')
    return file
    
def save_fits(path,name, data,hdr=None, ext_type=0, overwrite=True,norm=False):
    if norm==True:
        nan_mask = ~np.isfinite(data)
        masked = np.ma.masked_array(data, nan_mask)
        min, max = np.ma.min(masked), np.ma.max(masked)
        normed = (masked-min)/(max-min)
        output = np.where(normed.mask==True, np.nan, normed)
    else :
        output = data

    if ext_type == 0:
        ext = '.fits'
    elif ext_type == 1:
        ext = '.fit'
    fits.writeto(path+'/'+name+ext, output,header=hdr, overwrite=overwrite)
    #print(name + 'is/are saved at'+ path)

from astroquery.ipac.ned import Ned
from astroquery.simbad import Simbad
def radec(obj_name, catalog='ned'):
    if catalog == 'ned':
        tbl = Ned.query_object(obj_name)
        ra,dec = tbl['RA'][0],tbl['DEC'][0]
    elif catalog == 'simbad':
        tbl = Simbad.query_object(obj_name)
        ra,dec = tbl['ra'][0],tbl['dec'][0]
    return ra,dec

def prt_process(input):
    print(input + ' is/are done.')

from astropy.visualization import simple_norm
def norm(x, percent):
    return simple_norm(x, 'linear', percent=percent)

from astropy.wcs import WCS
from astropy.coordinates import get_icrs_coordinates
from astropy.nddata import Cutout2D

def im_crop(hdul, obj, offset=15, check_img=False, save_img=False, save_path=None):
    hdu, hdr = hdul.data, hdul.header#fits.getdata(path+'/sky_subed/coadd.fits', header=True)
    wcs = WCS(hdr)
    cen_coord = get_icrs_coordinates(obj)#SkyCoord(ra=ra, dec=dec,frame='icrs', unit='deg')
    x,y = wcs.world_to_pixel(cen_coord)#wcs.all_world2pix(ra, dec,0)#
    area = int(2048 - ((2*offset*60)/1.89))
    croped_hdu = Cutout2D(hdu, (x,y), area, wcs=wcs)
    croped = croped_hdu.data
    if check_img == True:
        fig, ax = plt.subplots(1,1,figsize=(8,8), subplot_kw=dict(projection=wcs))
        data = croped
        wcs = croped_hdu.wcs
        #print(new_hdu.header.cards)
        ax.imshow(data, norm=norm(data, percent=90), origin='lower')#, cmap='grey_r')
        ax.set_xlabel('R.A.')
        ax.set_ylabel('Decl.')
        plt.show()
    
    if (save_img == True) and (type(save_path)==str):
        new_hdu = fits.PrimaryHDU(croped, wcs.to_header())
        fits.writeto(save_path, new_hdu.data, new_hdu.header, overwrite=True)
