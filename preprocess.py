from pipeline.processing.frameproc import Master, Process
from astropy.io import fits
from pipeline.utils import file_list, mkdir, save_fits
import ray
import sys, time

path = '/Users/jang-in-yeong/240508/2'
obj = 'NGC5907'
ext_type = 1#.fit is 1, .fits is 0. default is 0(.fits)

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

ray.init(num_cpus=4)
@ray.remote
def mask(i,pix,amp_r, amp_mask=True):
    hdul = hdul_list[i]
    hdu = fits.getdata(hdul)
    process.mask(hdu,i,1.,pix,amp_r,amp_mask=amp_mask)
    time.sleep(0.1)

amp_mask = ray.put(True)
band = 'L'
if band == 'u':
    amp_mask=master.ampl_mask

works = [mask.remote(i,0.84,100,amp_mask) for i in range(len(hdul_list))]
while len(works):
    dones, works = ray.wait(works)
    ray.get(dones[0])
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
    save_fits(process.path+'/sky_subed',process.obj+'2_'+str(n),data=data,hdr=hdr,ext_type=process.ext_type)

ray.shutdown()
ray.init(num_cpus=4)
work = [bkg_sub.remote(pp_list,mask_list,i, 2) for i in range(len(pp_list))]
while len(work):
    done, work = ray.wait(work)
    ray.get(done[0])
ray.shutdown()

#astrometry.sh generate
process.astrometry(index_loc='~/solve/index4200',radius=1.5)
sys.exit()