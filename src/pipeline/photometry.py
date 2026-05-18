import sys, os
sys.path.append(os.path.abspath('./src/pipeline'))
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.stats import sigma_clipped_stats, sigma_clip
from scipy.optimize import curve_fit
from masking import region_mask
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
        

    def bkg_std(self,hdu, size=5,offset=15):
        std_list = []
        #median_list = []
        hdu_x, hdu_y = hdu.shape
        area = int(2048 - ((2*offset*60)/self.pix))
        print(area)
        mask = np.zeros_like(hdu)
        mask[int(hdu_y/2)-area//2:int(hdu_y/2)+area//2, int(hdu_x/2)-area//2:int(hdu_x/2)+area//2] = region_mask(hdu[int(hdu_y/2)-area//2:int(hdu_y/2)+area//2, int(hdu_x/2)-area//2:int(hdu_x/2)+area//2], 1, self.pix, ampglow=False)
        arr = np.ma.masked_where(mask, np.ma.masked_equal(hdu, 0))
        x,y = arr.shape
        center_x, center_y = int(x/2), int(y/2)
        ran_x, ran_y = [], []
        for i in range(1000):
            rand_st_x = np.random.randint(center_x-area//2, center_x+area//2-size)
            rand_st_y = np.random.randint(center_y-area//2, center_y+area//2-size)
            ran_x.append(rand_st_x)
            ran_y.append(rand_st_y)
            bin_arr = arr[rand_st_x:rand_st_x+size, rand_st_y:rand_st_y+size]
            std1 = np.ma.std(bin_arr)
            #mean, median1, std1 = sigma_clipped_stats(bin_arr, cenfunc='median', stdfunc='mad_std', sigma=3)
            #median_list.append(median1)
            std_list.append(std1)
        """
        mean, std_median, std = sigma_clipped_stats(np.array(std_list).astype(np.float32),
                                                cenfunc='median', stdfunc='mad_std', sigma=3.)
        """
        std_median = np.nanmedian(std_list)
        print(f'sigma={std_median}')
        self.bkg_noise = std_median
        """
        print(np.max(ran_x) - np.min(ran_x), np.max(ran_y)-np.min(ran_y))
        hist_arr = arr[int(hdu_y/2)-area//2:int(hdu_y/2)+area//2, int(hdu_x/2)-area//2:int(hdu_x/2)+area//2]
        hist_data = np.where(hist_arr.mask==True, np.nan, hist_arr.data)
        counts, bins = np.histogram(hist_data, bins=64, range=(-500,500))
        fig, ax = plt.subplots(1,3)
        ax[0].scatter(ran_x, ran_y, s=3)
        ax[0].set_xlabel('sampling_x')
        ax[0].set_ylabel('sampling_y')
        ax[1].imshow(arr, norm=norm(arr[int(hdu_y/2)-area//2:int(hdu_y/2)+area//2, int(hdu_x/2)-area//2:int(hdu_x/2)+area//2]), origin='lower')
        ax[2].bar(bins[:-1], counts, width=500/64)
        plt.show();sys.exit()
        """
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
            counts, bins = np.histogram(mM, bins=32)
            ax[0].bar(bins[:-1], counts, color='C1')
            ax[0].set_xlabel('M-m')
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

def sb_limit_proc(path, obj,file_name,pix,color=str):
    hdu = fits.open(path+'/sky_subed/'+file_name+'.fits')[0].data
    #mask = region_mask(hdu,0.8,pix,ampglow=False)
    phot = Phot(path, obj,file_name,pix)
    std_noise = phot.bkg_std(hdu,5,20)
    zp = phot.phot_stdz(color, plot=True)
    

sb_limit_proc('~/NGC5907','NGC5907','coadd',1.89,'r')