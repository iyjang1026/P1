import numpy as np
import matplotlib.pyplot as plt
import sys
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table
import sep

def fwhm_stats(path):
    hdu, hdr = fits.getdata(path, header=True)
    data =  hdu.astype(hdu.dtype.newbyteorder('='))
    bkg = sep.Background(data)
    obj,seg = sep.extract(data-bkg.back(), thresh=3, err=bkg.rms(), minarea=4, deblend_cont=1.0, segmentation_map=True)
    tbl = Table(obj)
    n_pix = tbl['npix']
    pa = sigma_clip(tbl['theta'][n_pix<30], cenfunc='median', stdfunc='mad_std', sigma=3)
    stbl = tbl[n_pix<30][pa.mask==False]
    eps = 1 - (stbl['b']/stbl['a'])
    median_fwhm = np.median(eps)#sigma_clip(eps, cenfunc='median', stdfunc='mad_std', sigma=3))
    """
    print(median_fwhm)
    plt.scatter(pa[pa.mask==False],eps, s=3)
    plt.axvline(x=np.ma.median(pa), linestyle='dashed', c='C1', label=f'median={np.ma.median(pa):.2f}')
    plt.axvline(x=np.ma.median(pa)+np.ma.std(pa), linestyle='dashed', c='C1')
    plt.axvline(x=np.ma.median(pa)-np.ma.std(pa), linestyle='dashed', c='C1')

    plt.axhline(y=np.ma.median(eps), linestyle='dashed', c='C1', label=f'median={np.ma.median(pa):.2f}')
    plt.axhline(y=np.ma.median(eps)+np.ma.std(eps), linestyle='dashed', c='C1')
    plt.axhline(y=np.ma.median(eps)-np.ma.std(eps), linestyle='dashed', c='C1')

    plt.xlabel('pa');plt.ylabel('eps')
    plt.show();sys.exit()
    """    
    return median_fwhm
"""
from pipeline.utils import file_list

files = file_list('/Users/jang-in-yeong/NGC5907/pp', ext_type=1)
fwhm_l = []
for i in range(len(files)):
    fwhm = fwhm_stats(files[i])
    fwhm_l.append(fwhm)
fwhm_arr = np.array(fwhm_l)
frac = len(fwhm_arr[fwhm_arr<=0.2])/len(fwhm_arr)
print(np.median(fwhm_arr))
print(frac)
x = np.arange(0,len(files),1)
plt.scatter(x, fwhm_arr, s=3)
plt.show()
sys.exit()
"""