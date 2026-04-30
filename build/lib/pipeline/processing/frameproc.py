import sys, os
#sys.path.append(os.path.abspath('./src/pipeline'))

from pipeline.utils import file_list, save_fits, radec
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import numpy as np

from pipeline.sky import poly_sky_model, rbf_sky_model
from pipeline.masking import region_mask, simple_masking

class Master:
    def __init__(self, path=str, ext_type=0):
        self.path = path 
        self.ext_type = ext_type

    def master_bias(self):
        bias_list = file_list(self.path+'/BIAS', self.ext_type)
        master_list = []
        for i in range(len(bias_list)):
            hdu = fits.open(bias_list[i])[0].data 
            master_list.append(hdu)
        mean, master_b, std = sigma_clipped_stats(np.array(master_list,dtype=np.float32),cenfunc='median',
                                                stdfunc='mad_std',sigma=3,axis=0)
        self.bias = master_b
        save_fits(self.path+'/process','master_bias', master_b,ext_type=self.ext_type)

    def master_dark(self):
        dark_list = file_list(self.path+'/DARK', self.ext_type)
        master_list = []
        for i in range(len(dark_list)):
            hdu = fits.open(dark_list[i])[0].data 
            master_list.append(hdu-self.bias)
        mean, master_d, std = sigma_clipped_stats(np.array(master_list,dtype=np.float32),cenfunc='median',
                                                stdfunc='mad_std',sigma=3,axis=0)
        save_fits(self.path+'/process','master_dark', master_d,ext_type=self.ext_type)
        self.dark = master_d
        #return master_d
    
    def amp_mask(self):
        mask = simple_masking(self.dark)
        self.ampl_mask = mask

    def master_flat(self):
        db_list = file_list(self.path+'/db_subed', self.ext_type)
        mask_list = file_list(self.path+'/mask', self.ext_type)
        if len(db_list) != len(mask_list):
            raise ValueError(f'db_list and mask_list are not same size!! {len(db_list), len(mask_list)}')
        scl_list = []
        m_list = []
        for i in range(len(db_list)):
            hdu = fits.open(db_list[i])[0].data
            mask = fits.open(mask_list[i])[0].data
            tmp_flat = np.ma.masked_where(mask,hdu)
            medianf = np.ma.median(tmp_flat)
            scaled = np.ma.masked_array((tmp_flat - medianf)/medianf, dtype=np.float16)
            scl_list.append(scaled)
            m_list.append(medianf.astype(np.float16))
        median1 = np.median(m_list)
        mean, sc_flat,std = sigma_clipped_stats(np.ma.masked_array(scl_list,dtype=np.float16)
                                                ,cenfunc='median',stdfunc='mad_std',sigma_lower=6,
                                                sigma_upper=3,axis=0)
        master_f = np.array(sc_flat * median1 + median1, dtype=np.float32)
        save_fits(self.path+'/process','master_flat', master_f,ext_type=self.ext_type)
        self.flat = master_f
        #return master_f

class Process:
    def __init__(self, path=str,obj=str, ext_type=0):
        self.path = path
        self.obj = obj
        self.ext_type = ext_type

    def db_sub(self, bias=np.ndarray, dark=np.ndarray):
        l_list = file_list(self.path+'/LIGHT',self.ext_type)
        for i in range(len(l_list)):
            n = format(i,'04')
            hdl = fits.open(l_list[i])[0]
            hdr = hdl.header
            hdu = hdl.data
            db_subed = hdu.astype(np.float32) - bias - dark
            save_fits(self.path+'/db_subed','db_subed'+str(n),db_subed,hdr,ext_type=self.ext_type)

    def mask(self,hdu=np.ndarray,i=int,pix=float,amp_r=bool|np.ndarray,amp_mask=True):
        #hdu = fits.getdata(hdul)
        mask = region_mask(hdu, 0.99,pix,disk_r=amp_r,ampglow=amp_mask)
        n = format(i,'04')
        save_fits(self.path+'/mask','mask_'+str(n),data=mask,ext_type=self.ext_type)

    def proc(self,db_list, flat=np.ndarray):
        mean, median, std = sigma_clipped_stats(flat, cenfunc='median',
                                                stdfunc='mad_std',sigma=3)
        for i in range(len(db_list)):
            hdu = fits.open(db_list[i])[0].data
            hdr = fits.open(db_list[i])[0].header
            pp_img = hdu / (flat/median)
            n = format(i, '04')
            save_fits(self.path+'/pp','pp_'+self.obj+str(n),data=pp_img,hdr=hdr,ext_type=self.ext_type)

    def sky_sub(self,pp_list, mask_list,i=int,order=2,model='polynomial',bin=64):
        if len(pp_list) != len(mask_list):
            raise ValueError(f'pp_list and mask_list are not same size!! {len(pp_list),len(mask_list)}')
        hdu = fits.open(pp_list[i])[0].data 
        hdr = fits.open(pp_list[i])[0].header
        mask = fits.open(mask_list[i])[0].data
        m_data = np.ma.masked_array(hdu, mask, dtype=np.float32)
        bkg=None
        if model == 'polynomial':
            bkg = np.array(poly_sky_model(m_data,bin,order=order), dtype=np.float32)
        elif model == 'rbf':
            if bin > 16:
                raise ValueError("bin must be smaller than 16 at rbf modeling")
            else:
                bkg = np.array(rbf_sky_model(m_data, bin), dtype=np.float32)
        
        subed = np.array(hdu-bkg).astype(np.float32)
        hdr.append(('sky_sub', 'Python', 'sky subtraction' ))
        return subed, hdr
        
    def astrometry(self, index_loc=None|str, radius=float):
        ext_type = self.ext_type
        if ext_type == 0:
            ext = '.fits'
        else :
            ext = '.fit'
        ra,dec = radec(self.obj)
        file = open(self.path+'/'+self.obj+'.sh', 'w')
        if type(index_loc) == str:
            file.write(f'solve-field --index-dir {index_loc} --use-source-extractor -3 {ra} -4 {dec} -5 {radius} --no-plots {self.obj}*{ext}')
        else:    
            file.write(f'solve-field --use-source-extractor -3 {ra} -4 {dec} -5 {radius} --no-plots {self.obj}*{ext}')
        
        file.close()

