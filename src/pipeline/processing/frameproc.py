import sys, os, site, time
from utils import file_list, save_fits, radec, mkdir, norm
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, sigma_clip
from scipy.stats import mode
import numpy as np
import matplotlib.pyplot as plt

from sky import poly_sky_model, rbf_sky_model
from masking import region_mask, simple_masking, se_mask

class Master:
    def __init__(self, path=str, ext_type=0):
        self.path = path 
        self.ext_type = ext_type

    def master_bias(self):
        bias_list = file_list(self.path+'/BIAS', self.ext_type)
        master_list = []
        for i in range(len(bias_list)):
            hdu = fits.getdata(bias_list[i])
            master_list.append(hdu)
        median_b= np.ma.median(sigma_clip(np.array(master_list,dtype=np.float32),cenfunc='median',
                                                stdfunc='mad_std',sigma=3,axis=0), axis=0)
        
        master_b = np.where(median_b.mask==False, median_b, np.nan)
        self.bias = master_b
        save_fits(self.path+'/process','master_bias', master_b,ext_type=self.ext_type)

    def master_dark(self):
        dark_list = file_list(self.path+'/DARK', self.ext_type)
        master_list = []
        for i in range(len(dark_list)):
            hdu = fits.getdata(dark_list[i])
            master_list.append(hdu-self.bias)
        median_d= np.ma.median(sigma_clip(np.array(master_list,dtype=np.float32),cenfunc='median',
                                                stdfunc='mad_std',sigma=3,axis=0),axis=0)
        master_d = np.where(median_d.mask==True, np.nan, median_d)
        save_fits(self.path+'/process','master_dark', master_d,ext_type=self.ext_type)
        self.dark = master_d
        #return master_d
    
    def amp_mask(self, threshold):
        mask = simple_masking(self.dark, detect_threshold=threshold)
        save_fits(self.path+'/process', 'amp_mask', mask, ext_type=self.ext_type)
        self.ampl_mask = mask

    def scaling_func(self,x, scale='median'):
        if scale == 'median':
            return np.ma.median(x)
        elif scale == 'mode':
            return mode(x)[0]
        else :
            raise IndexError(f'scale functions are not selected.')
        
    def master_flat(self, hsigma=3, lsigma=3, scale='median'):
        db_list = file_list(self.path+'/db_subed', self.ext_type)
        mask_list = file_list(self.path+'/mask', self.ext_type)
        if len(db_list) != len(mask_list):
            raise ValueError(f'db_list and mask_list are not same size!! {len(db_list), len(mask_list)}')
        scl_list = []
        #m_list = []
        for i in range(len(db_list)):
            hdu = fits.getdata(db_list[i])
            mask = fits.getdata(mask_list[i])
            tmp_flat = np.ma.masked_array(hdu, mask)
            #sc_parm = self.scaling_func(tmp_flat, scale=scale)
            clipped = sigma_clip(tmp_flat, cenfunc='median', stdfunc='mad_std', sigma=3)
            if scale == 'median':
                sc_parm = np.ma.median(clipped)
            elif scale == 'mode':
                sc_parm = mode(clipped[clipped.mask==False])[0].astype(np.float16)
            scaled = np.ma.masked_array((tmp_flat)/sc_parm,mask=mask, dtype=np.float16)
            scl_list.append(scaled)
            #m_list.append(sc_parm.astype(np.float16))
        #median1 = np.median(m_list)
        sc_flat = np.ma.median(sigma_clip(np.ma.masked_array(scl_list,dtype=np.float16)
                                                ,cenfunc='median',stdfunc='mad_std',sigma_lower=lsigma,
                                                sigma_upper=hsigma, axis=0),axis=0)
        master_f = np.array(sc_flat, dtype=np.float32)#* median1 + median1, dtype=np.float32)
        hdr = fits.PrimaryHDU(master_f).header
        card = fits.header.Card('Method', scale, 'master flat scaling method')
        hdr.append(card)
        save_fits(self.path+'/process',f'master_flat_{scale}', master_f,hdr=hdr,ext_type=self.ext_type)
        self.flat = master_f
        self.fhdr = hdr
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
            hdu, hdr = fits.getdata(l_list[i],header=True)
            db_subed = hdu.astype(np.float32) - bias - dark
            save_fits(self.path+'/db_subed','db_subed'+str(n),db_subed,hdr,ext_type=self.ext_type)
            time.sleep(0.1)

    def mask(self,hdu=np.ndarray,threshold=float,pix=float,amp_r=bool|np.ndarray,amp_mask=True,ellipse_mask=True, ell_num=20):
        #hdu = fits.getdata(hdul)
        mask = region_mask(hdu, threshold,pix,disk_r=amp_r,ampglow=amp_mask,ellipse_mask=ellipse_mask, ell_num=ell_num)
        return mask

    def se_mask(self,hdu=np.ndarray,threshold=float,pix=float,amp_r=bool|np.ndarray,amp_mask=True, ellipse_mask=True, ell_num=20):
        #hdu = fits.getdata(hdul)
        mask = se_mask(hdu,threshold=threshold,pix=pix,disk_r=amp_r,ampglow=amp_mask, ellipse_mask=ellipse_mask, ell_num=ell_num)
        return mask

    def proc(self,db_list, flat,f_hdr, norm=False):
        #m_flat = np.ma.masked_where((flat==np.nan)|(flat==0), flat)
        
        #median = np.ma.median(sigma_clip(m_flat, cenfunc='median', stdfunc='mad_std',sigma=3))
        
        for i in range(len(db_list)):
            hdu, hdr = fits.getdata(db_list[i], header=True)
            nan_mask = np.ma.masked_array(hdu, ~np.isfinite(hdu))
            pp_img = nan_mask / flat
            output = np.where(pp_img.mask==True, np.nan, pp_img)
            n = format(i, '04')
            hdr.append(f_hdr.cards['METHOD'])
            save_fits(self.path+'/pp','pp_'+self.obj+str(n),data=output,hdr=hdr,ext_type=self.ext_type, norm=norm)
            time.sleep(0.1)

    def sky_sub(self,pp_list, mask_list,i=int,order=2,model='polynomial',bin=64):
        if len(pp_list) != len(mask_list):
            raise ValueError(f'pp_list and mask_list are not same size!! {len(pp_list),len(mask_list)}')
        hdu, hdr = fits.getdata(pp_list[i], header=True)
        mask = fits.getdata(mask_list[i])
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
        hdr.append(('sky_sub', str(bin), 'sky subtraction' ))
        return subed, hdr
        
    def astrometry(self, index_loc=None, radius=float):
        ext_type = self.ext_type
        if ext_type == 0:
            ext = '.fits'
        else :
            ext = '.fit'
        ra,dec = radec(self.obj)
        file = open(self.path+'/'+self.obj+'.sh', 'w')
        if index_loc != None:
            file.write(f'solve-field --index-dir {index_loc} --use-source-extractor -3 {ra} -4 {dec} -5 {radius} --no-plots {self.obj}*{ext}\nrm -rf *.new *.xyls *.rdls *.corr *.axy *.solved *.match')
        else:    
            file.write(f'solve-field --use-source-extractor -3 {ra} -4 {dec} -5 {radius} --no-plots {self.obj}*{ext}\nrm -rf *.xyls *.rdls *.corr *.axy *.solved *.match')
        
        file.close()    