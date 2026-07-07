from pipeline.processing.frameproc import Master, Process
from astropy.io import fits
from pipeline.utils import file_list, mkdir, save_fits
import ray
import sys, time
import numpy as np
from scipy.stats import mode
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats


start_time = time.time()
path = '/Users/jang-in-yeong/NGC5907'
obj = 'NGC5907'
ext_type = 1#.fit is 1, .fits is 0. default is 0(.fits)

master = Master(path,ext_type)
process = Process(path, obj, ext_type)

#bias, dark subtraction and amplifier glow masking
mkdir(path, 'process')
master.master_bias()
master.master_dark()
#master.amp_mask(threshold=6.)

mkdir(path,'db_subed')
process.db_sub(bias=master.bias, dark=master.dark)

#masking and master flat
mkdir(path, 'mask')
hdul_list = file_list(process.path + '/db_subed', ext_type=process.ext_type)

ray.init(num_cpus=4)
@ray.remote
def mask(i,pix,amp_r, amp_mask=True):
    hdul = hdul_list[i]
    hdu = fits.getdata(hdul)
    mask = process.mask(hdu,2.,pix,amp_r,amp_mask=amp_mask,ellipse_mask=True, ell_num=25)
    n = format(i,'04')
    save_fits(process.path+'/mask','mask_'+str(n),data=mask,ext_type=process.ext_type)
    #time.sleep(0.1)

amp_mask = ray.put(True)
band = 'L'
if band == 'u':
    amp_mask=master.ampl_mask

works = [mask.remote(i,1.89,150,amp_mask) for i in range(len(hdul_list))]
while len(works):
    dones, works = ray.wait(works)
    ray.get(dones[0])
ray.shutdown()

#flat-fielding
master.master_flat(scale='median')

db_list = file_list(process.path+'/db_subed', ext_type=process.ext_type)
flat, fhdr = fits.getdata(process.path+'/process/master_flat_median.fit', header=True)#master.flat, master.fhdr #

mkdir(process.path, 'pp')
process.proc(db_list, flat, f_hdr=fhdr, norm=False)
endtime = time.time()
print(f'eta={endtime-start_time}')
#sys.exit()

#sky subtraction
pp_list = file_list(process.path+'/pp', process.ext_type)

mkdir(process.path, 'sky_mask')
ray.init(num_cpus=4)
@ray.remote
def mask(i,pix,amp_r, amp_mask=True):
    hdul = pp_list[i]
    hdu = fits.getdata(hdul)
    mask = process.mask(hdu,0.5,pix,amp_r,amp_mask=amp_mask,ellipse_mask=True, ell_num=25)
    n = format(i,'04')
    save_fits(process.path+'/sky_mask','mask_'+str(n),data=mask,ext_type=process.ext_type)

amp_mask = ray.put(True)
band = 'L'
if band == 'u':
    amp_mask=master.ampl_mask

works = [mask.remote(i,1.89,150,amp_mask) for i in range(len(pp_list))]
while len(works):
    dones, works = ray.wait(works)
    ray.get(dones[0])
ray.shutdown()

bin=16
pwd = mkdir(path, 'sky_subed')
mask_list = file_list(process.path + '/sky_mask', process.ext_type)

@ray.remote
def bkg_sub(pp_list, mask_list, i, order, bkg_type='median'):
    hdu, hdr = fits.getdata(pp_list[i], header=True)
    mask = fits.getdata(mask_list[i])
    masked = np.ma.masked_array(hdu, mask)
    clipped = sigma_clip(masked, cenfunc='median', stdfunc='mad_std', sigma=3)
    if bkg_type == 'median':
        bkg_const= np.ma.median(clipped)
        data = hdu - bkg_const
        hdr.append(('bkg_type', bkg_type, 'Background subtraction type'))
    elif bkg_type == 'mean':
        bkg_const= np.ma.mean(clipped)
        data = hdu - bkg_const
        hdr.append(('bkg_type', bkg_type, 'Background subtraction type'))
    elif bkg_type == 'mode':
        bkg_const= mode(clipped[clipped.mask==False])[0]#
        data = hdu - bkg_const
        hdr.append(('bkg_type', bkg_type, 'Background subtraction type'))
    elif bkg_type == 'polynomial':
        data, hdr = process.sky_sub(pp_list,mask_list,i, order, bin=bin)
        hdr.append(('bkg_type', bkg_type+str(order), 'Background subtraction type'))
    else :
        raise ValueError('That type is not availiable')
    n = format(i, '04')
    save_fits(pwd,process.obj+str(n),data=data.astype(np.float32),hdr=hdr,ext_type=process.ext_type,norm=False)

ray.init(num_cpus=4)
work = [bkg_sub.remote(pp_list,mask_list,i, 2, bkg_type='median') for i in range(len(pp_list))]
while len(work):
    done, work = ray.wait(work)
    ray.get(done[0])
ray.shutdown()
endtime = time.time()
print(f'eta={endtime-start_time}')


#astrometry.sh generate
process.astrometry(index_loc='~/solve/index4200',radius=1.5, use_scamp=True)
endtime = time.time()
print(f'eta={start_time-start_time}')
sys.exit()