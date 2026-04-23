import numpy as np
from astropy.io import fits
from utils import file_list, mkdir, save_fits,prt_process
from frameproc import Master, Process
from masking import simple_masking
import ray
import sys

def full_proc(path,obj,filter=str, ext_type=int):
    master = Master(path,ext_type)
    process = Process(path, obj, ext_type)
    
    #bias, dark subtraction and amplifier glow masking
    mkdir(path, 'process')
    master.master_bias()
    master.master_dark()
    master.amp_mask()

    mkdir(path,'db_subed')
    process.db_sub(bias=master.bias, dark=master.dark)

    #masking and master flat
    mkdir(path, 'mask')
    hdul_list = file_list(process.path + '/db_subed', ext_type=process.ext_type)
    
    @ray.remote
    def mask(hdul,i,pix,amp_r, amp_mask=True):
        hdu = fits.getdata(hdul)
        process.mask(hdu,i,pix,amp_r,amp_mask=amp_mask)
    
    #for i in range(len(hdul_list)):
    #    hdul = hdul_list[i]
    #    process.mask(hdul, i, pix=1.89, amp_r=300, amp_mask=master.ampl_mask)
    if filter == 'u':
        amp_mask=master.ampl_mask
    else :
        amp_mask = True

    ray.init(num_cpus=6)
    ray.get([mask.remote(hdul_list[i],i,pix=1.89,amp_r=300,amp_mask=amp_mask) for i in range(len(hdul_list))])
    ray.shutdown()
    
    master.master_flat()

    #flat-fielding
    mkdir(path,'pp')
    db_list = file_list(process.path+'/db_subed', ext_type=process.ext_type)
    process.proc(db_list, master.flat)
    
    #sky subtraction
    mkdir(path, 'sky_subed')
    pp_list = file_list(process.path+'/pp', process.ext_type)
    mask_list = file_list(process.path + '/mask', process.ext_type)
    @ray.remote
    def bkg_sub(pp_list, mask_list, i):
        data, hdr = process.sky_sub(pp_list,mask_list,i)
        n = format(i, '04')
        save_fits(process.path+'/sky_subed',process.obj+'_'+str(n),data=data,hdr=hdr,ext_type=process.ext_type)
    
    ray.init(num_cpus=6)
    ray.get([bkg_sub.remote(pp_list,mask_list,i) for i in range(len(pp_list))])
    ray.shutdown()

    #astrometry.sh generate
    process.astrometry(radius=1.5)
    sys.exit()

if __name__ == "__main__":
    path = '/volumes/ssd/u_test'    
    full_proc(path,'NGC784',filter='u',ext_type=1)