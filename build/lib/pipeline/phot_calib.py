import sys, os
sys.path.append(os.path.abspath('./src/pipeline'))
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.stats import sigma_clipped_stats, sigma_clip
from astropy.modeling import models, fitting
from scipy.stats import mode
from scipy.optimize import curve_fit
from masking import region_mask
from utils import radec
from astropy.visualization import simple_norm
import warnings
warnings.filterwarnings('ignore')

def norm(x):
    return simple_norm(x, 'linear', percent=99)

def stdz_mag(count,z_p,a):
        #mag = -2.5*np.log10(count) + z_p
        return a*count+z_p#mag

class Phot:
    def __init__(self, path, obj,file_name, pix):
        self.path = path
        self.obj = obj
        self.file_name = file_name
        self.pix = pix
        self.data = Table.read(path+'/sky_subed/'+self.file_name+'.cat', format='ascii', converters={'obsid':str})
        self.sdss = Table.read(path + '/sdss_'+obj+'.csv', format='ascii') #check!! 
        

    def bkg_std(self,hdul,frame_size=2048, offset=15, plot=False):
        hdu = hdul.data
        hdr = hdul.header
        wcs = WCS(hdr)
        ra, dec = radec(self.obj)
        cen_coord = SkyCoord(ra, dec, frame='fk5', unit='deg')
        x, y = wcs.world_to_pixel(cen_coord)
        std_list = []
        size = int(10/self.pix)
        area = int(frame_size - ((2*offset*60)/self.pix))
        
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
        #print(std_array)
        std_median = np.nanmedian(std_array)
        print(f'sigma={std_median}')
        self.bkg_noise = std_median
        if plot == True:
            #print(np.max(ran_x) - np.min(ran_x), np.max(ran_y)-np.min(ran_y))
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
        sdsscat = sdss['ra', 'dec', 'g','r','u', 'Err_r', 'Err_g', 'Err_u']
        objcat = data['ALPHAPEAK_J2000','DELTAPEAK_J2000','FLUX_BEST', 'MAGERR_BEST']#, 'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE']
        #obj_cat = objcat[(objcat['ERRAWIN_IMAGE']<0.01)&(objcat['ERRBWIN_IMAGE']<0.01)]
        sdss_coord = SkyCoord(ra=sdsscat['ra'], dec=sdsscat['dec'],unit='deg', frame='fk5')
        obj_coord = SkyCoord(ra=objcat['ALPHAPEAK_J2000'], dec=objcat['DELTAPEAK_J2000'],unit='deg', frame='fk5')

        idx1, d2d1, d3d1 = sdss_coord.match_to_catalog_sky(obj_coord)
        sdss_data = sdsscat
        obj_f = objcat[idx1]

        obj_flux = obj_f['FLUX_BEST']
        
        sdss_mag = sdss_data[color]
        z_m = -2.5*np.log10(np.array(obj_flux))
        sdss_mag = np.array(sdss_mag)
        u = sdss_data['u']
        g = sdss_data['g']
        r = sdss_data['r']

        #m = -2.5*np.log10(count)
        zm = sigma_clip(z_m, cenfunc='median', stdfunc='mad_std', sigma=2)
        m = z_m[zm.mask==False]
        mag = sdss_mag[zm.mask==False]
        mM = mag - m
        
        z =  sigma_clip(mM, cenfunc='median', stdfunc='mad_std',sigma=3)
        obj_magerr = obj_f['MAGERR_BEST'][zm.mask==False]
        sdss_magerr = sdss_data[str('Err_'+color)][zm.mask==False]
        mag_err = obj_magerr[z.mask==False]
        sdss_err = sdss_magerr[z.mask==False]
        saturated = mag[z.mask==True]

        """
        r1 = r[z.mask==False]
        g1 = g[z.mask==False]
        u1 = u[z.mask==False]

        if color == 'r':
            c = r1
            c_refer = g1
        elif color == 'g':
            c = g1
            c_refer = r1
        else :
            c = u1
            c_refer = g1
        """
        print(f'# of detected stars = {len(z_m)}')
        print(f'Fitted star fraction = {len(mag[z.mask==False])/len(sdss_mag):.4f}')
        print(f'Saturated star fraction = {len(saturated)/len(sdss_mag):.4f}')
        
        t_r = m[z.mask==False]
        z0 = np.ma.median(z)
        fit = fitting.LinearLSQFitter()
        l_init = models.Linear1D(slope=1, intercept=z0)
        fitted_line = fit(l_init, t_r, mag[z.mask==False])
        a = np.array(fitted_line.slope)
        zp = np.array(fitted_line.intercept)
        """
        def std_formular(count1, zp,a):
            return a*count1+zp #-2.5*np.log10(count1) + z1

        popt,pcov = curve_fit(std_formular,t_r,mag[z.mask==False],p0=[z0,1], sigma=np.ma.std(m[z.mask==False]), maxfev=1000)
        zp = popt[0]
        """
        #a,z0 = np.median(alpha*(l1-l2)), zp
        sb_lim = zp - 2.5*np.log10(1*self.bkg_noise/(self.pix*10))
        print(f'Z_p = {zp}')
        print(f'a = {a}')
        print(f'1sigma SB Limit = {sb_lim}')

        if plot == True:
            fig,ax = plt.subplots(1,2)
            bins=32
            #counts, bins = np.histogram(mM[~np.isnan(mM)], bins=32)
            width=(np.max(bins)-np.min(bins))/32
            ax[0].hist(mM, bins=bins, color='C1')#, range=[27.7,28.3])#, width=width)
            ax[0].set_xlabel('$M - m$')
            ax[0].set_ylabel('# of stars')
            ax[0].axvline(x=zp, linestyle='dashed', linewidth=2, c='grey')

            ax[1].scatter(10**(-0.4*z_m),sdss_mag,s=2,c='grey')
            ax[1].scatter(10**(-0.4*t_r),mag[z.mask==False],s=2,c='r')
            #ax[1].scatter(m,mag,s=2,c='grey')
            #ax[1].scatter(t_r,c,s=2,c='r')
            m.sort()
            x = np.linspace(np.min(z_m), np.max(z_m), len(z_m))
            ax[1].plot(10**(-0.4*x),fitted_line(x),c='k',linewidth=1.5)

            ax[1].set_xscale('log', base=10)
            
            ax[1].set_xlabel('Flux(log10)')
            ax[1].set_ylabel(f'$\mu_{color}$')    
            ax[1].text(np.min(10**(-0.4*z_m))+10, np.min(sdss_mag), f'$Z_p$ = {zp:.2f}'+'\n$\mu_{limit,1\sigma}$'+f' = {sb_lim:.2f}', bbox={'boxstyle':'square', 'fc':'white'})
            ax[1].set_title(f'{color}-band SB limit of {self.obj}')
            plt.show()
        
        return zp,a, sb_lim, len(z_m)



def sb_limit_proc(path, obj,file_name,pix,frame_size,offset,color=str, plot=False):
    hdul = fits.open(path+'/sky_subed/'+file_name+'.fits')[0]
    hdr = hdul.header
    exptime = np.array(hdr['EXPTIME'], dtype=np.float32)
    phot = Phot(path, obj,file_name,pix)
    std_noise = phot.bkg_std(hdul, frame_size,offset, plot=plot)
    zp,a, sb_lim, num_star = phot.phot_stdz(color, plot=plot)

"""
    return exptime, std_noise, zp, a, sb_lim, num_star

data_list = [['obj','exptime', 'std_noise', 'zp', 'a', 'sb_lim', 'num_star']]    
ssd_path = '/volumes/ssd/intern/25_summer'
obj_list = ['M101','M51','NGC6946','NGC4236', 'NGC5907']
file_list = ['M101_L', 'M51_L', 'NGC6946_L','NGC4236_r','NGC5907_r']
for i in range(len(file_list)):
    exptime, std_noise, zp, a, sb_lim, num_star =sb_limit_proc(ssd_path+'/'+file_list[i], obj_list[i], 'coadd', 1.89, 2048 ,20,'r', plot=False)
    data_list.append([obj_list[i],exptime, std_noise, zp, a, sb_lim, num_star])

import csv
f = open('/Users/jang-in-yeong/phot_data.csv', 'w',newline='', encoding='ascii')
writer = csv.writer(f)
writer.writerows(data_list)
"""