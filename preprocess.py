from pipeline.processing.frameproc import Master, Process
from astropy.io import fits
from pipeline.utils import file_list, mkdir, save_fits
import ray
import sys

path = '/volumes/USB128/2026-05-05/g'
obj = 'M51'
ext_type = 0 #.fit is 1, .fits is 0. default is 0(.fits)

master = Master(path,ext_type)
process = Process(path, obj, ext_type)

#bias, dark subtraction and amplifier glow masking
mkdir(path, 'process')
master.master_bias()
master.master_dark()
#master.amp_mask()

mkdir(path,'db_subed')
process.db_sub(bias=master.bias, dark=master.dark)

#masking and master flat
mkdir(path, 'mask')
hdul_list = file_list(process.path + '/db_subed', ext_type=process.ext_type)

@ray.remote
def mask(hdul,i,pix,amp_r, amp_mask=False):
    hdu = fits.getdata(hdul)
    process.mask(hdu,i,1.5,pix,amp_r,amp_mask=amp_mask)

amp_mask = False
band = 'L'
if band == 'u':
    amp_mask=master.ampl_mask

ray.shutdown()
ray.init(num_cpus=6)
ray.get([mask.remote(hdul_list[i],i,pix=1.86,amp_r=300,amp_mask=amp_mask) for i in range(len(hdul_list))])
ray.shutdown()

#flat-fielding
master.master_flat()
mkdir(path,'pp')
db_list = file_list(process.path+'/db_subed', ext_type=process.ext_type)

process.proc(db_list, master.flat)

#sky subtraction
mkdir(path, 'sky_subed')
pp_list = file_list(process.path+'/pp', process.ext_type)
mask_list = file_list(process.path + '/mask', process.ext_type)
@ray.remote
def bkg_sub(pp_list, mask_list, i, order):
    data, hdr = process.sky_sub(pp_list,mask_list,i, order)
    n = format(i, '04')
    save_fits(process.path+'/sky_subed',process.obj+'_'+str(n),data=data,hdr=hdr,ext_type=process.ext_type)

ray.shutdown()
ray.init(num_cpus=6)
ray.get([bkg_sub.remote(pp_list,mask_list,i, order=2) for i in range(len(pp_list))])
ray.shutdown()

#astrometry.sh generate
process.astrometry(index_loc='~/solve/index4200',radius=1.5)
sys.exit()