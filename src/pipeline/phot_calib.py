import sys, os
import numpy as np
from scipy.stats import skew
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from masking import region_mask, se_mask
from utils import radec
from astropy.visualization import simple_norm
from cpnn import Variable
import cpnn.functions as F
from cpnn import Function
import warnings
warnings.filterwarnings('ignore')

def norm(x):
    return simple_norm(x, 'linear', percent=99)

def stdz_mag(count,z_p,a):
        #mag = -2.5*np.log10(count) + z_p
        return a*count+z_p#mag

def optimizer(mag_inst, mag_cat, err_inst, z0):
    x = Variable(mag_inst)
    y = mag_cat
    W1 = Variable(np.array(0.5))
    b1 = Variable(np.array(z0))

    def predict(x):
        y = F.linear(x, W1, b1)
        return y

    lr = 0.2
    iters = 10000

    class ChiSquaredError(Function):
        def __init__(self, sigma):
            self.sigma = sigma
        def forward(self, x0, x1):
            diff = ((x0 - x1)**2)/(self.sigma**2)
            y = diff.sum() #(diff**2).sum() /len(diff)
            return y

        def backward(self, gy):
            x0, x1 = self.inputs
            diff = x0 - x1
            gx0 = gy * diff * (2 / self.sigma**2)
            gx1 = -gx0
            return gx0, gx1
        
    def chi_squared_error(x0, x1):
        return ChiSquaredError(err_inst)(x0, x1)
        
    for i in range(iters):
        y_pred = predict(x)
        loss = chi_squared_error(y, y_pred)#

        W1.cleargrad()
        b1.cleargrad()
        loss.backward()
        W1.data -= lr * W1.grad.data
        b1.data -= lr * b1.grad.data
    return W1.data, b1.data

class Phot():
    def __init__(self, path, obj,file_name, pix):
        self.path = path
        self.obj = obj
        self.file_name = file_name
        self.pix = pix
        self.data = Table.read(path+'/sky_subed/'+self.file_name+'.cat', format='ascii', converters={'obsid':str})
        self.sdss = Table.read(path + '/sdss_'+obj+'.csv', format='ascii') #check!! 
        self.hdu, self.hdr = fits.getdata(path+'/sky_subed/'+file_name+'.fits', header=True)

    def bkg_std(self,threshold=0.5, ell_num=4, frame_size=2048,offset=15,max_pix=1024,frac=0.7,iter=2000,pos_return=False, plot=False):
        hdu = self.hdu
        hdr = self.hdr
        wcs = WCS(hdr)
        ra, dec = radec(self.obj)
        #cen_coord = get_icrs_coordinates(self.obj)#SkyCoord(ra=ra, dec=dec,frame='icrs', unit='deg')
        x,y = wcs.all_world2pix(ra, dec,0)#wcs.world_to_pixel(cen_coord)#
        std_list = []
        median_list = []
        area = int(frame_size - ((2*offset*60)/self.pix))
        size = int(np.sqrt(max_pix/frac))
        
        croped = hdu[int(y)-area//2:int(y)+area//2, int(x)-area//2:int(x)+area//2]
        mask = np.ones_like(hdu)
        mask[int(y)-area//2:int(y)+area//2, int(x)-area//2:int(x)+area//2] = region_mask(croped, threshold, self.pix, ampglow=False, ellipse_mask=True, ell_num=ell_num)
        arr = np.ma.masked_array(hdu, mask) #np.where(mask!=0, np.nan, hdu)#np.ma.masked_where(mask, np.ma.masked_equal(hdu, 0))
        ran_x, ran_y = [], []
        np.random.seed(0)
        while len(std_list)<iter:
            rand_st_x = np.random.randint(x-area//2, x+area//2-size)
            rand_st_y = np.random.randint(y-area//2, y+area//2-size)
            bin_arr = arr[rand_st_y:rand_st_y+size, rand_st_x:rand_st_x+size]
            if bin_arr[bin_arr.mask==0].size/bin_arr.size >=frac:
                std1 = np.ma.std(sigma_clip(bin_arr, cenfunc='median', stdfunc='mad_std', sigma=3))
                median1 = np.ma.median(sigma_clip(bin_arr, cenfunc='median', stdfunc='mad_std', sigma=3))
                std_list.append(std1)
                median_list.append(median1)
                ran_x.append(rand_st_x)
                ran_y.append(rand_st_y)
        std_array = np.array(std_list)
        median_array = np.array(median_list)
        std_median = np.median(std_array)
        pos_func = np.array(np.array(ran_x)/2048 + 2*np.array(ran_y)/4096)
        print(f'sigma={std_median}')
        print(f'sampling sigma={np.ma.std(sigma_clip(median_array, cenfunc='median', stdfunc='mad_std', sigma=3))}')
        print(f'total sigma={np.ma.std(sigma_clip(arr, cenfunc='median', stdfunc='mad_std', sigma=3))}')
        self.bkg_noise = std_median
        if plot == True:
            hist_arr = arr[int(y)-area//2:int(y)+area//2, int(x)-area//2:int(x)+area//2]
            counts, bins = np.histogram(hist_arr, bins=64, range=(-500,500))
            width = (np.max(bins)-np.min(bins))/64
            fig, ax = plt.subplots(1,3, figsize=(18,5))
            ax[0].scatter(np.array(ran_x)+(size//2), np.array(ran_y)+(size//2), s=3, c='tomato')
            ax[0].imshow(arr, norm=norm(hist_arr), origin='lower')
            ax[0].set_xlim(int(x)-area//2-10,int(x)+area//2+10)
            ax[0].set_ylim(int(y)-area//2-10,int(y)+area//2+10)
            ax[0].set_title('Sampling point distribution')

            #ax[1].bar(bins[:-1], counts, width=width, color='C0')
            #norm_arr = (hist_arr - np.nanmean(hist_arr))/np.nanstd(hist_arr)
            clipped = sigma_clip(hist_arr, cenfunc='median', stdfunc='mad_std')
            ax[1].hist(hist_arr[hist_arr.mask==0], bins=32, color='C0', range=(np.ma.min(clipped)*1.5, np.ma.max(clipped)*1.5), label=str(skew(hist_arr[hist_arr.mask==0])))
            ax[1].axvline(x=np.median(hist_arr[hist_arr.mask==0]), linestyle='dashed', c='C1', label=f'median={np.median(hist_arr[hist_arr.mask==0]):.2f}')
            ax[1].set_title(f'Total $\sigma=${np.ma.std(sigma_clip(arr, cenfunc='median', stdfunc='mad_std', sigma=3)):.2f}')
            ax[1].set_xlabel('Pixel value')
            ax[1].set_ylabel('count')
            ax[1].legend()
            #norm_median = (median_array - np.nanmean(median_array))/np.nanstd(median_array)
            ax[2].hist(median_array, bins=32, color='C0', label=str(skew(median_array)))
            ax[2].axvline(x=np.median(median_array), linestyle='dashed', c='C1', label=f'median={np.median(median_array):.2f}')
            ax[2].set_title(f'Sampling $\sigma=${np.ma.std(sigma_clip(median_array, cenfunc='median', stdfunc='mad_std', sigma=3)):.2f}')
            ax[2].set_xlabel('Pixel value')
            ax[2].set_ylabel('count')
            ax[2].legend()
            plt.show()
            
        
        if pos_return == True:
            return std_median, np.ma.std(pos_func)
        else:
            return std_median

    def phot_stdz(self,color, plot=False):
        data = self.data
        sdss = self.sdss
        #extract coordinate
        sdsscat = sdss['ra', 'dec', 'g','r','u', 'Err_r', 'Err_g', 'Err_u']#[sdss[color]>12]
        objcat = data['ALPHAPEAK_J2000','DELTAPEAK_J2000','FLUX_BEST', 'FLUXERR_BEST']#[data['CLASS_STAR']>=0.75]
        sdss_coord = SkyCoord(ra=sdsscat['ra'], dec=sdsscat['dec'],unit='deg', frame='fk5')
        obj_coord = SkyCoord(ra=objcat['ALPHAPEAK_J2000'], dec=objcat['DELTAPEAK_J2000'],unit='deg', frame='fk5')

        idx1, d2d1, d3d1 = sdss_coord.match_to_catalog_sky(obj_coord)
        sdss_data = sdsscat
        obj_f = objcat[idx1]

        obj_flux = obj_f['FLUX_BEST']
        fluxerr = obj_f['FLUXERR_BEST']
        sdss_mag = sdss_data[color]
        m = -2.5*np.log10(np.array(obj_flux))
        mag = np.array(sdss_mag)
        mM = mag - m
        
        z =  sigma_clip(mM, cenfunc='median', stdfunc='mad_std',sigma=3)
        t_r = m[z.mask==False]
        z0 = np.ma.median(z)
        """
        obj_magerr = obj_f['MAGERR_BEST'][zm.mask==False]
        sdss_magerr = sdss_data[str('Err_'+color)][zm.mask==False]
        mag_err = obj_magerr[z.mask==False]
        sdss_err = sdss_magerr[z.mask==False]
        """
        saturated = mag[z.mask==True]

        u = sdss_data['u']#[zm.mask==False]
        g = sdss_data['g']#[zm.mask==False]
        r = sdss_data['r']#[zm.mask==False]
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
        
        print(f'# of detected stars = {len(m)}')
        print(f'Fitted star fraction = {len(mag[z.mask==False])/len(sdss_mag):.4f}')
        print(f'Saturated star fraction = {len(saturated)/len(sdss_mag):.4f}')
        
        a, zp = optimizer(t_r,mag[z.mask==False],fluxerr[z.mask==False],z0)
        def std_formular(count1,zp,a):
            return a*count1+zp
        sb_lim = zp - 2.5*np.log10(1*self.bkg_noise/(self.pix*10))
        print(f'Z_p = {zp}')
        print(f'a = {a}')
        print(f'1sigma SB Limit = {sb_lim}')
        #flux = 10**(-.4*(t_r))
        #mag_err = abs(-2.5*(self.bkg_noise/(flux*np.log(10))))
        if plot == True:
            fig,ax = plt.subplots(1,2, figsize=(11,5))
            bins=32
            #counts, bins = np.histogram(mM[~np.isnan(mM)], bins=32)
            width=(np.max(bins)-np.min(bins))/32
            ax[0].hist(mM, bins=bins, color='C1')#, range=[27.7,28.3])#, width=width)
            ax[0].set_xlabel('$M - m$')
            ax[0].set_ylabel('# of stars')
            ax[0].axvline(x=zp, linestyle='dashed', linewidth=2, c='grey')
            ax[1].scatter(a*m+zp,sdss_mag,s=3,c='grey', label='saturated')
            ax[1].scatter(a*t_r+zp,mag[z.mask==False],s=2,c='r', label='non-saturated')
            ax[1].legend()
            x = np.linspace(-100, 100, len(m))
            def line(x):
                return x
            ax[1].plot(x,line(x),c='k',linewidth=1.5, linestyle='dashed', label='fitted')
            #ax[1].set_xscale('log', base=10)
            ax[1].set_xlabel('$L_{calib}$')
            ax[1].set_ylabel(f'$\mu_{color}$')    
            ax[1].set_ylim(20, 10)
            ax[1].set_xlim(20,10)
            ax[1].text(np.ma.min(a*m+zp)+3, np.max(sdss_mag)-0.5, f'$Z_p$ = {zp:.2f}'+'\n$\mu_{limit,1\sigma}$'+f' = {sb_lim:.2f}', bbox={'boxstyle':'square', 'fc':'white'})
            ax[1].set_title(f'{color}-band Calib of {self.obj}')
            plt.show()
            
        
        return a, zp, sb_lim, len(m)



def sb_limit_proc(path, obj,file_name,pix,frame_size,offset,color=str,bkg_plot=False, plot=False):
    phot = Phot(path, obj,file_name,pix)
    exptime = np.array(phot.hdr['EXPTIME'], dtype=np.float32)
    std_noise = phot.bkg_std(frame_size=frame_size,offset=offset, max_pix=1024,frac=0.6,iter=2000, plot=bkg_plot)
    a, zp,sb_lim, num_star = phot.phot_stdz(color, plot=plot)

    #return exptime, std_noise,a, zp, sb_lim, num_star

"""
path = '~/NGC5907'
sb_limit_proc(path, 'NGC5907', 'coadd', 1.89, 2048, 10, 'r',bkg_plot=True, plot=True)
"""
"""
data_list = [['obj','exptime', 'std_noise', 'a','zp', 'sb_lim', 'num_star']]    
ssd_path = '/volumes/ssd/intern/25_summer'
obj_list = ['M101','M51','NGC6946','NGC4236', 'NGC5907']
file_list = ['M101_L', 'M51_L', 'NGC6946_L','NGC4236_r','NGC5907_r']
for i in range(len(file_list)):
    exptime, std_noise,a, zp, sb_lim, num_star =sb_limit_proc(ssd_path+'/'+file_list[i], obj_list[i], 'coadd', 1.89, 2048 ,20,'r', plot=False)
    data_list.append([obj_list[i],exptime, std_noise,a, zp, sb_lim, num_star])

import csv
f = open('/Users/jang-in-yeong/phot_data_ver2.csv', 'w',newline='', encoding='ascii')
writer = csv.writer(f)
writer.writerows(data_list)
"""