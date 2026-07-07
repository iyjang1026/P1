import numpy as np
import sys
import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve
from photutils.segmentation import detect_sources, make_2dgaussian_kernel, SourceCatalog
from photutils.isophote import EllipseGeometry, Ellipse, build_ellipse_model
from photutils.aperture import EllipticalAperture
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
import sys
from masking import obj_rej_mask, se_mask
from utils import radec
from astropy.convolution import convolve
from astropy.stats import sigma_clip

def norm(x, percent=99):
    return simple_norm(x, 'linear', percent=percent)

class EllipseIsophote:
    def __init__(self, path, obj_name, pix_scale=1.89):
        self.path = path
        self.obj = obj_name
        self.pix = pix_scale
    
    def data_imp(self, file=None):
        if file != None:
            hdu, hdr = fits.getdata(file, header=True)
        else:
            hdu = fits.getdata(self.path+'/sky_subed/coadd.fits')
            hdr = fits.getheader(self.path+'/sky_subed/coadd.fits')
        self.hdu = hdu #np.ma.masked_where(hdu>49000,hdu)
        #plt.imshow(self.hdu, origin='lower');plt.show();sys.exit()
        self.hdr = hdr

    def obj_mask(self, threshold=list, iter=int,saturate=False,d_pix=200, ell_mask=True, ell_num=10):
        ra, dec = radec(self.obj)
        if len(threshold)==1:
                threshold1 = threshold*iter
        else :
            threshold1 = threshold
        if type(saturate)!=bool:
            mask = np.where(self.hdu>=saturate, 1,0)
        else:
            mask=np.zeros_like(self.hdu)
        for i in range(iter):
            """
            masked, threshold, aper = se_mask(self.hdu, threshold=threshold1[i],obj_rejc=True,pix=self.pix,
                                               hdr=self.hdr, obj_name=self.obj,npix=4,
                                               kernel_size=1,ellipse_mask=ell_mask,ell_num=ell_num, bkg_thrsh=True)
            """
            masked, threshold, aper = obj_rej_mask(self.hdu, threshold1[i], hdr=self.hdr,ra=ra, dec=dec,
                                                   npix=4,kernel_size=1,mask=mask,deblend_pix=d_pix,
                                                   ellipse_mask=ell_mask, ell_num=ell_num, bkg_thrsh=True)
            
            mask1 = masked+mask
        self.mask = mask1
        self.threshold = threshold
        self.aper = aper

    def exp_param(self,npix=10, plot=False):
        wcs = WCS(self.hdr)
        ra, dec = radec(self.obj)
        coord = SkyCoord(ra=ra, dec=dec,frame='fk5', unit='deg')
        x, y = wcs.world_to_pixel(coord)
        cen_x, cen_y = int(x), int(y)
        #print(cen_x, cen_y);sys.exit()
        self.cenx = cen_x
        self.ceny = cen_y
        data = np.ma.masked_array(data=self.hdu, mask=self.mask)
        i0 = np.ma.median(data[cen_y-1:cen_y+1,cen_x-1:cen_x+1])
        exp = np.exp(1)
        h_intens = i0 / exp
        kernel = make_2dgaussian_kernel(3.,9)
        conv_arr = convolve(data-h_intens, kernel=kernel)
        #plt.imshow(conv_arr,norm=norm(conv_arr), origin='lower');plt.show();sys.exit()
        seg_map = detect_sources(conv_arr, np.nanstd(conv_arr), n_pixels=npix)
        cat = SourceCatalog(conv_arr, seg_map)
        xcen,ycen = cat.x_centroid,cat.y_centroid
        idx = np.where((np.min(abs(x-xcen))==abs(x-xcen))&(np.min(abs(y-ycen)==abs(y-ycen))))
        cat1 = cat[idx][0]
     
        h, pa, ell = 3*cat1.semimajor_axis.value, cat1.orientation.value+90, cat1.ellipticity.value
        self.geo = EllipseGeometry(xcen[0], ycen[0], h, ell, (pa-90)*np.pi/180)
        self.h = h
        self.pa = pa
        self.ell = ell
        print(pa, ell, i0, h)

        if plot == True:
            pos = (cat1.x_centroid, cat1.y_centroid)
            aper = EllipticalAperture(pos, a=3*cat1.semimajor_axis.value, b=3*cat1.semiminor_axis.value, theta=cat1.orientation.value*np.pi/180)
            plt.imshow(seg_map, origin='lower')
            aper.plot(color='r', linewidth=0.7)
            plt.show()

        return pa, ell, i0, h

    def sersic_param(self,npix=2000, plot=False):
        data = np.ma.masked_array(self.hdu, self.mask)
        wcs = WCS(self.hdr)
        ra, dec = radec(self.obj)
        coord = SkyCoord(ra=ra, dec=dec,frame='fk5', unit='deg')
        x, y = wcs.world_to_pixel(coord)
        #cen_x, cen_y = int(x), int(y)

        aper = self.aper
        area = aper.area
        aper_sum, sum_err = aper.do_photometry(data)
        i_e = aper_sum[0]/(2*area)
        kernel = make_2dgaussian_kernel(3.,9)
        conv_arr = convolve(data-i_e, kernel=kernel)
        seg_map = detect_sources(conv_arr, np.ma.std(conv_arr), n_pixels=npix)
        cat = SourceCatalog(conv_arr, seg_map)
        
        xcen,ycen = cat.x_centroid,cat.y_centroid
        idx = np.where((np.min(abs(x-xcen))==abs(x-xcen))&(np.min(abs(y-ycen)==abs(y-ycen))))
        cat1 = cat[idx][0]

        r_e, pa, ell = 3*cat1.semimajor_axis.value, cat1.orientation.value+90, cat1.ellipticity.value
        self.geo = EllipseGeometry(xcen[0], ycen[0], r_e, ell, (pa-90)*np.pi/180)
        self.h = r_e
        self.pa = pa
        self.ell = ell
        print(pa, ell, 4, i_e, r_e)

        if plot == True:
            pos = (cat1.x_centroid, cat1.y_centroid)
            aper = EllipticalAperture(pos, a=3*cat1.semimajor_axis.value, b=3*cat1.semiminor_axis.value, theta=cat1.orientation.value*np.pi/180)
            plt.imshow(seg_map, origin='lower')
            aper.plot(color='r', linewidth=0.7)
            plt.show()
        return pa, ell, 4, i_e, r_e
    
    def d25_param(self, npix=200, plot=False):
        data = np.ma.masked_array(self.hdu, self.mask)
        wcs = WCS(self.hdr)
        ra, dec = radec(self.obj)
        coord = SkyCoord(ra=ra, dec=dec,frame='fk5', unit='deg')
        x, y = wcs.world_to_pixel(coord)

        i_25 = 166.6
        kernel = make_2dgaussian_kernel(3.,9)
        conv_arr = convolve(data-i_25, kernel=kernel)
        seg_map = detect_sources(conv_arr, np.ma.std(conv_arr), n_pixels=npix)
        cat = SourceCatalog(conv_arr, seg_map)
        
        xcen,ycen = cat.x_centroid,cat.y_centroid
        idx = np.where((np.min(abs(x-xcen))==abs(x-xcen))&(np.min(abs(y-ycen)==abs(y-ycen))))
        cat1 = cat[idx][0]

        r_e, pa, ell = 3*cat1.semimajor_axis.value, cat1.orientation.value+90, cat1.ellipticity.value
        self.geo = EllipseGeometry(xcen[0], ycen[0], r_e, ell, (pa-90)*np.pi/180)
        self.h = r_e
        self.pa = pa
        self.ell = ell
        print(pa, ell, 4, i_25, r_e)

        if plot == True:
            pos = (cat1.x_centroid, cat1.y_centroid)
            aper = EllipticalAperture(pos, a=3*cat1.semimajor_axis.value, b=3*cat1.semiminor_axis.value, theta=cat1.orientation.value*np.pi/180)
            init_aper = self.aper
            plt.imshow(seg_map, origin='lower')
            aper.plot(color='r', linewidth=0.7)
            init_aper.plot(color='r', linewidth=0.7)
            plt.show()
        return pa, ell, 4, i_25, r_e

    def ellipse(self,integrmode='bilinear',step=0.1,sigma=3.,fflag=0.7,minsma=1.5, fix_pa=False, fix_center=True, fix_eps=False,linear=False, modeling=True, conv=False):    
        img = np.ma.masked_array(self.hdu, self.mask)#np.ma.masked_where((self.mask==1), self.hdu)#
        #plt.imshow(img, origin='lower');plt.show();sys.exit()
        ellipse = Ellipse(img, geometry=self.geo)
        isolist = ellipse.fit_image(sma0=self.h, minsma=minsma,integrmode=integrmode,
                                    linear=linear,step=step, sclip=sigma, n_clip=5, fflag=fflag,
                                    fix_pa=fix_pa, fix_center=fix_center, fix_eps=fix_eps) #isophote #maxsma=5*self.h,self.h*0.001
        tbl = isolist.to_table()    
        print(tbl)
        if modeling==True:
            model = build_ellipse_model(img.shape, isolist) #modeling
            if conv == True:
                conv_model = convolve(model, make_2dgaussian_kernel(3, (5,5)))
                return conv_model, tbl
            else:
                return model, tbl
        else :
            return tbl
"""
path = '~/2026-05-14'
obj = 'NGC5797'

fig, ax = plt.subplots(1,3, figsize=(15,5), gridspec_kw={'width_ratios':[1,1,1]})
ellipse = EllipseIsophote(path, obj)
ellipse.data_imp(file=path+'/psf_sub.fits')
ellipse.obj_mask(threshold=[1.], iter=1, ell_mask=False)
#plt.imshow(np.ma.masked_array(ellipse.hdu, ellipse.mask), norm=norm(ellipse.hdu), origin='lower');plt.show();sys.exit()
#ellipse.d25_param(plot=True);sys.exit()
ellipse.sersic_param(npix=200,plot=False);sys.exit()
model,tbl = ellipse.ellipse(step=0.05, fflag=0.5, modeling=True)
radius = tbl['sma'] * 1.89 #np.sqrt((-(tbl['ellipticity']-1)/tbl['sma'])) * 1.89
intens = tbl['intens'] / (1.89**2)
mag = -2.5*np.log10(intens)*1.005+29.1
#tbl.write(path+'/iso_tbl_'+obj+'.csv', format='ascii.csv', overwrite=True)
#fits.writeto(path+'/model.fits', model, overwrite=True)
ax[0].scatter(radius, mag, s=2)
ax[1].imshow(np.log10(model), origin='lower')
ax[2].imshow(ellipse.hdu-model, norm=norm(ellipse.hdu, percent=90), origin='lower')
#sys.exit()
ax[0].set_xlabel('sma(arcsec)')
ax[0].set_ylabel('$\mu_r$')
ax[0].invert_yaxis()
#plt.legend()
plt.show()
"""