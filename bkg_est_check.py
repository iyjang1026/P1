import sys, warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode, skew

from astropy.io import fits
from astropy.stats import sigma_clipped_stats, sigma_clip

from pipeline.utils import file_list, norm
from pipeline.sky import poly_sky_model
from pipeline.masking import se_mask, region_mask
from photutils.aperture import RectangularAperture

warnings.filterwarnings('ignore')

def hist_check(path):
    files = file_list(path+'/pp', ext_type=1)
    masks = file_list(path+'/sky_mask', ext_type=1)
    
    mean_l = []
    median_l = []
    std_l = []
    percen_l = []
    #contrast = []
    step = 1
    for i in range(0,len(files),step):
        data = np.ma.masked_invalid(fits.getdata(files[i]))#
        mask = se_mask(data.data, 0.5, 1.89, ampglow=True, ellipse_mask=True, ell_num=25)#fits.getdata(masks[i]) #
        masked = np.ma.masked_array(data, mask)
        clipped = sigma_clip(masked, cenfunc='median', stdfunc='mad_std', sigma_upper=3, sigma_lower=3)
        const_bkg = np.ma.mean(clipped)
        
        #print(clipped);sys.exit()
        #bkg = poly_sky_model(np.ma.masked_array(data-const_bkg, mask), 16, 2)
        #bkg = poly_sky_model(masked, 16, 2)
        
        #plt.imshow(masked, norm=norm(masked, percent=90), origin='lower');plt.colorbar();plt.show();sys.exit()

        output = np.ma.masked_array(data-const_bkg,mask)
        median, std = np.ma.median(output), np.ma.std(output)
        """
        plt.hist(output[output.mask==False], bins=64, range=(median-6*std, median+6*std))
        plt.axvline(x=np.ma.median(output), linestyle='dashed', linewidth=1, c='C1', label=f'median={np.ma.median(output):.2f}')
        plt.axvline(x=np.ma.mean(output), linestyle='dotted', linewidth=1, c='r', label=f'mean={np.ma.mean(output):.2f}')
        plt.legend()
        plt.xlabel('pixel value')
        plt.ylabel("count(log10)")
        plt.yscale('log', base=10)
        plt.title(f'median={median}')
        plt.show();sys.exit()
        """
        mean, median, std = np.ma.mean(output), np.ma.median(output), np.ma.std(output)
        h_percentile = np.percentile(output[output.mask==False], 75)
        l_percentile = np.percentile(output[output.mask==False], 25)
        percen_l.append((abs(h_percentile-median)-abs(median-l_percentile))/(np.ma.max(output)-np.ma.min(output)))
        #contrast.append(skew(output[output.mask==False]))#np.ma.min(data)/np.ma.max(data))
        mean_l.append(mean)
        median_l.append(median)
        std_l.append(std)
    
    x = np.arange(0, len(files), step=step)
    #print(len(x), np.array(percen_l));sys.exit()
    fig, ax = plt.subplots(3,1, sharex=True)
    ax[0].scatter(x, np.array(mean_l),s=3, label='mean', c='C0')
    ax[0].scatter(x, np.array(median_l), s=3, label='median', c='C1')
    ax[1].scatter(x, np.array(std_l), s=3, label='std', c='C2')
    ax[2].scatter(x, np.array(percen_l), s=3, label='skewness', c='C3')
    #ax[2].scatter(x, np.array(contrast), s=3, label='contrast', c='C4')
    fig.supxlabel('frame num')
    ax[0].set_ylabel('ADU')
    fig.legend()
    fig.suptitle('median sub')
    plt.show()

#hist_check('/Users/jang-in-yeong/NGC5907')

from astropy.wcs import WCS
from astropy.coordinates import get_icrs_coordinates
from pipeline.masking import se_mask, region_mask

def bkg_std(path,obj,offset=15,max_pix=1024,frac=0.7,iter=2000,pos_return=False, plot=False):
    hdu, hdr = fits.getdata(path, header=True)
    wcs = WCS(hdr)
    #ra, dec = radec(self.obj)
    cen_coord = get_icrs_coordinates(obj)#SkyCoord(ra=ra, dec=dec,frame='icrs', unit='deg')
    x,y = wcs.world_to_pixel(cen_coord)#wcs.all_world2pix(ra, dec,0)#
    """
    box = RectangularAperture((x,y), 634*2, 634*2)
    fig, ax = plt.subplots(1,1,subplot_kw=dict(projection=wcs))
    ax.imshow(hdu, norm=norm(hdu, percent=90), origin='lower')
    box.plot(color='r', ax=ax)
    ax.set_xlabel('R.A.')
    ax.set_ylabel('Decl.')
    plt.show();sys.exit()
    """
    std_list = []
    median_list = []
    area = int(2048 - ((2*offset*60)/1.89))
    size = int(np.sqrt(max_pix/frac))
    
    croped = hdu[int(y)-area//2:int(y)+area//2, int(x)-area//2:int(x)+area//2]
    mask = np.ones_like(hdu)
    mask[int(y)-area//2:int(y)+area//2, int(x)-area//2:int(x)+area//2] = region_mask(croped, .3, 1.89, ampglow=False, ellipse_mask=True, ell_num=20)
    arr = np.ma.masked_array(hdu, mask) #np.where(mask!=0, np.nan, hdu)#np.ma.masked_where(mask, np.ma.masked_equal(hdu, 0))
    #plt.imshow(arr, norm=norm(arr, percent=90), origin='lower');plt.show();sys.exit()
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
    
    if plot == True:
        hist_arr = arr[int(y)-area//2:int(y)+area//2, int(x)-area//2:int(x)+area//2]
        fig, ax = plt.subplots(1,3, figsize=(18,5))
        ax[0].scatter(np.array(ran_x)+(size//2), np.array(ran_y)+(size//2), s=3, c='tomato')
        ax[0].imshow(arr, norm=norm(hist_arr, percent=90), origin='lower')
        ax[0].set_xlim(int(x)-area//2-10,int(x)+area//2+10)
        ax[0].set_ylim(int(y)-area//2-10,int(y)+area//2+10)
        ax[0].set_title('Sampling point distribution')
        clipped = sigma_clip(hist_arr, cenfunc='median', stdfunc='mad_std', sigma=3)
        ax[1].hist(hist_arr[hist_arr.mask==0], bins=64, color='C0', label=str(skew(hist_arr[hist_arr.mask==0])))
        ax[1].axvline(x=np.ma.median(clipped), linestyle='dashed', c='C1', label=f'median={np.ma.median(clipped):.2f}')
        ax[1].set_title(f'Total $\sigma=${np.ma.std(sigma_clip(arr, cenfunc='median', stdfunc='mad_std', sigma=3)):.2f}')
        ax[1].set_xlabel('Normalized ADU')
        ax[1].set_ylabel('count')
        ax[1].legend()
        ax[1].set_yscale('log', base=10)

        clipped_hist = sigma_clip(median_array, cenfunc='median', stdfunc='mad_std')
        ax[2].hist(clipped_hist[clipped_hist.mask==False],bins=64, color='C0')
        #ax[2].axhline(y=np.median(median_array), linestyle='dashed', c='C1', label=f'median={np.median(median_array):.2f}')
        ax[2].axvline(x=np.ma.median(clipped_hist), linestyle='dashed', c='C1', label=f'median={np.ma.median(clipped_hist):.2f}')
        ax[2].set_title(f'Sampling $\sigma=${np.ma.std(clipped_hist):.2f}')
        ax[2].set_xlabel('median')
        ax[2].set_ylabel('position')
        #ax[2].set_yscale('log', base=10)
        ax[2].legend()
        plt.show()
    
    if pos_return == True:
        return std_median, np.ma.std(pos_func)
    else:
        return std_median

#bkg_std('~/NGC5907/sky_subed/coadd.fits', 'NGC5907', offset=10,frac=0.6, max_pix=1024, plot=True)

def img_hist(path, i=0):
    hdu = fits.getdata(path+'/pp/pp_NGC5907'+format(i,'04')+'.fit')
    mask = fits.getdata(path+'/sky_mask/mask_'+format(i,'04')+'.fit')
    data = np.ma.masked_array(hdu, mask)
    clipped = sigma_clip(data, cenfunc='median', stdfunc='mad_std', sigma=3)
    h_percentile = np.percentile(clipped[clipped.mask==False], 95)
    l_percentile = np.percentile(clipped[clipped.mask==False], 5)
    #plt.imshow(clipped,norm=norm(clipped, percent=90), origin='lower');plt.colorbar();plt.show();sys.exit()
    mean, median, std = np.ma.mean(clipped), np.ma.median(clipped), np.ma.std(clipped)
    print(abs(mean-median))
    skewness = (abs(h_percentile-median)-abs(median-l_percentile))/(np.ma.max(clipped)-np.ma.min(clipped))
    plt.hist(data[data.mask==False], bins=128, range=(median-6*std, median+6*std), label=f'{skewness}')
    plt.axvline(x=median, linestyle='dashed', linewidth=1, c='C1', label=f'median={median:.2f}')
    plt.axvline(x=mean, linestyle='dashed', linewidth=1, c='C3', label=f'mean={mean:.2f}')
    #plt.yscale('log', base=10)
    plt.title(f'{i}')
    plt.legend()
    plt.show()
"""
for i in range(0,360, 1):
    img_hist('~/NGC5907', i)
"""
"""
def weight_img(path, obj):
    hdu, hdr = fits.getdata(path+'/coadd.fits', header=True)
    weight = fits.getdata(path+'/coadd.weight.fits')
    mask = np.where(weight>=np.max(weight), 0, 1)
    obj_mask = region_mask(np.ma.masked_array(hdu, mask), 0.3, 1.89, ampglow=False)
    wcs = WCS(hdr)
    cen_coord = get_icrs_coordinates(obj)
    x,y = wcs.world_to_pixel(cen_coord)
    arr = np.ma.masked_array(hdu, obj_mask)
    std_list = []
    median_list = []
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

    plt.imshow(np.ma.masked_array(hdu, obj_mask), norm=norm(hdu, percent=90), origin='lower')
    plt.show()
    sys.exit()

#weight_img('~/NGC5907/sky_subed')
"""