import sys, os
sys.path.append(os.path.abspath('./src/pipeline'))
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.stats import sigma_clipped_stats, sigma_clip
from scipy.optimize import curve_fit
from masking import region_mask
from utils import radec
from astropy.visualization import simple_norm

import warnings
warnings.filterwarnings('ignore')

def norm(x):
    return simple_norm(x, 'linear', percent=99)

def stdz_mag(count,z_p):
        mag = -2.5*np.log10(count) + z_p
        return mag

class Phot:
    def __init__(self, path, obj,file_name, pix):
        self.path = path
        self.obj = obj
        self.file_name = file_name
        self.pix = pix
        self.data = Table.read(path+'/sky_subed/'+self.file_name+'.cat', format='ascii', converters={'obsid':str})
        self.sdss = Table.read(path + '/sdss_'+obj+'.csv', format='ascii') #check!! 
        

    def bkg_std(self,hdul,frame_size=2048, size=5,offset=15, plot=False):
        hdu = hdul.data
        hdr = hdul.header
        wcs = WCS(hdr)
        ra, dec = radec(self.obj)
        cen_coord = SkyCoord(ra, dec, frame='fk5', unit='deg')
        x0,y0 = hdu.shape
        x, y = wcs.world_to_pixel(cen_coord)
        #x,y = int(x0/2),int(y0/2)
        std_list = []
        #median_list = []
        area = int(frame_size - ((2*offset*60)/self.pix))
        #print(area)
        croped = hdu[int(y)-area//2:int(y)+area//2, int(x)-area//2:int(x)+area//2]
        mask = np.zeros_like(hdu)
        mask[int(y)-area//2:int(y)+area//2, int(x)-area//2:int(x)+area//2] += region_mask(croped, 1, self.pix, ampglow=False)
        arr = np.where(mask!=0, np.nan, hdu)#np.ma.masked_where(mask, np.ma.masked_equal(hdu, 0))
        ran_x, ran_y = [], []
        #for i in range(1000):
        while len(std_list)<2000:
            rand_st_x = np.random.randint(x-area//2, x+area//2-size)
            rand_st_y = np.random.randint(y-area//2, y+area//2-size)
            bin_arr = arr[rand_st_y:rand_st_y+size, rand_st_x:rand_st_x+size]
            if len(bin_arr[np.isnan(bin_arr)]) <1:
                std1 = np.nanstd(bin_arr)
                #mean, median1, std1 = sigma_clipped_stats(bin_arr, cenfunc='median', stdfunc='mad_std', sigma=3)
                #median_list.append(median1)
                
                std_list.append(std1)
                ran_x.append(rand_st_x)
                ran_y.append(rand_st_y)
                #print(len(std_list), std1)  
        """
        mean, std_median, std = sigma_clipped_stats(np.array(std_list).astype(np.float32),
                                                cenfunc='median', stdfunc='mad_std', sigma=3.)
        """
        std_array = np.array(std_list)
        print(std_array)
        std_median = np.nanmedian(std_array)
        print(f'sigma={std_median}')
        self.bkg_noise = std_median
        if plot == True:
            print(np.max(ran_x) - np.min(ran_x), np.max(ran_y)-np.min(ran_y))
            hist_arr = arr[int(y)-area//2:int(y)+area//2, int(x)-area//2:int(x)+area//2]
            #hist_data = np.where(hist_arr.mask==True, np.nan, hist_arr.data)
            counts, bins = np.histogram(hist_arr, bins=64, range=(-500,500))
            width = (np.max(bins)-np.min(bins))/64
            fig, ax = plt.subplots(1,2)
            ax[0].scatter(ran_x, ran_y, s=3, c='tomato')
            ax[0].imshow(arr, norm=norm(hist_arr), origin='lower')
            ax[1].bar(bins[:-1], counts, width=width, color='C0')
            ax[1].axvline(x=np.nanmedian(hist_arr), linestyle='dashed', c='C1')
            ax[1].axvline(x=np.nanmedian(hist_arr)+std_median, linestyle='dotted', c='C1')
            ax[1].axvline(x=np.nanmedian(hist_arr)-std_median, linestyle='dotted', c='C1')
            plt.show()
        
        return std_median

    def phot_stdz(self,color, plot=False):
        data = self.data
        sdss = self.sdss
        #extract coordinate
        sdsscat = sdss['ra', 'dec', 'g','r','u']
        objcat = data['ALPHAPEAK_J2000','DELTAPEAK_J2000','FLUX_BEST']#, 'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE']
        #obj_cat = objcat[(objcat['ERRAWIN_IMAGE']<0.01)&(objcat['ERRBWIN_IMAGE']<0.01)]
        sdss_coord = SkyCoord(ra=sdsscat['ra'], dec=sdsscat['dec'],unit='deg', frame='fk5')
        obj_coord = SkyCoord(ra=objcat['ALPHAPEAK_J2000'], dec=objcat['DELTAPEAK_J2000'],unit='deg', frame='fk5')

        idx1, d2d1, d3d1 = sdss_coord.match_to_catalog_sky(obj_coord)
        sdss_data = sdsscat
        obj_f = objcat[idx1]

        obj_flux = obj_f['FLUX_BEST']
        sdss_mag = sdss_data[color]
        count = np.array(obj_flux)
        mag = np.array(sdss_mag)
        u = sdss_data['u']
        g = sdss_data['g']
        r = sdss_data['r']

        m = -2.5*np.log10(count)
        mM = mag - m
        
        z =  sigma_clip(mM, cenfunc='median', stdfunc='mad_std', sigma=3)
        r1 = r[z.mask==False]
        g1 = g[z.mask==False]
        u1 = u[z.mask==False]
        saturated = mag[z.mask==True]
        print(f'Fitted star fraction = {len(u1)/len(mag)}')
        print(f'Saturated star fraction = {len(saturated)/len(mag)}')

        if color == 'r':
            c = r1
            l1 = g1
            l2 = r1
        elif color == 'g':
            c = g1
            l1 = g1
            l2 = r1
        else :
            c = u1
            l1 = u1
            l2 = g1
        
        def std_formular(count1,z1):
            return -2.5*np.log10(count1) + z1
        
        t_r = count[z.mask==False]
        popt,pcov = curve_fit(std_formular,t_r,c)
        zp = popt[0]
        
        #a,z0 = np.median(alpha*(l1-l2)), zp
        
        if plot == True:
            sb_lim = zp - 2.5*np.log10(1*self.bkg_noise/(self.pix*10))
            print(f'Z_p is {zp}')
            print(f'SB Limit is {sb_lim}')
            
            def line(x,a,b):
                return a*(-2.5*np.log10(x)) +b
            
            popt_line,pcov_line = curve_fit(line,t_r,stdz_mag(t_r, zp))
            fig,ax = plt.subplots(1,2)
            counts, bins = np.histogram(mM[~np.isnan(mM)], bins=32)
            width=(np.max(bins)-np.min(bins))/32
            ax[0].bar(bins[:-1], counts, color='C1', width=width)
            ax[0].set_xlabel('$M - m$')
            ax[0].set_ylabel('# of stars')
            ax[0].axvline(x=zp, linestyle='dashed', linewidth=2, c='grey')

            ax[1].scatter(count,mag,s=2,c='grey')
            ax[1].scatter(t_r,c,s=2,c='r')

            count.sort()
            ax[1].plot(count,line(count,*popt_line),c='k',linewidth=1.5)

            ax[1].set_xscale('log', base=10)
            ax[1].set_xlabel('Flux(log10)')
            ax[1].set_ylabel(f'$\mu_{color}$')    
            ax[1].text(10**3.5, 10, f'$Z_p$ = {zp:.2f}'+'\n$\mu_{limit,1\sigma}$'+f' = {sb_lim:.2f}', bbox={'boxstyle':'square', 'fc':'white'})
            ax[1].set_title(f'{color}-band SB limit of {self.obj}')
            plt.show()

        return zp

def sb_limit_proc(path, obj,file_name,pix,frame_size,size,offset,color=str):
    hdul = fits.open(path+'/sky_subed/'+file_name+'.fits')[0]
    #mask = region_mask(hdu,0.8,pix,ampglow=False)
    #plt.imshow(np.ma.masked_array(hdu, mask));plt.show();sys.exit()
    phot = Phot(path, obj,file_name,pix)
    std_noise = phot.bkg_std(hdul, frame_size,size,offset, plot=True)
    zp = phot.phot_stdz(color, plot=True)
    

#sb_limit_proc('~/NGC5907', 'NGC5907', 'coadd', 1.89, 2048, 5,10,'r') # ('~/M51', 'M51', 'coadd', 0.84, 3008, int(10/0.84), 15, 'r') #

