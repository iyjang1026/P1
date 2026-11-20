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
    bkg = sep.Background(data.astype(np.float32))
    obj,seg = sep.extract(data-bkg.back(), thresh=3, err=bkg.rms(), minarea=4, deblend_cont=1.0, segmentation_map=True)
    tbl = Table(obj)
    n_pix = tbl['npix']
    pa = sigma_clip(tbl['theta'][n_pix<30], cenfunc='median', stdfunc='mad_std', sigma=3)
    stbl = tbl[n_pix<30][pa.mask==False]
    eps = 1 - (stbl['b']/stbl['a'])
    median_eps = np.median(eps)#sigma_clip(eps, cenfunc='median', stdfunc='mad_std', sigma=3))
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
    return median_eps

from pipeline.utils import file_list
def multi_qual(path, thresh=0.2):
    files = file_list(path, ext_type=1)
    eps_l = []
    for i in range(len(files)):
        eps = fwhm_stats(files[i])
        eps_l.append(eps)
    eps_arr = np.array(eps_l)
    idx = np.where(eps_arr>=thresh)[0]+1
    print(idx)
    #np.savetxt(path+'/rejected_idx.txt', idx, fmt='%d', delimiter=',')
    #return eps_arr[eps_arr<thresh]
    frac = len(eps_arr[eps_arr<=thresh])/len(eps_arr)
    print(np.median(eps_arr))
    print(frac)
    x = np.arange(0,len(files),1)
    plt.scatter(x, eps_arr, s=3)
    plt.show()
    sys.exit()

#multi_qual('/Users/jang-in-yeong/NGC784/r2/LIGHT', 0.17)