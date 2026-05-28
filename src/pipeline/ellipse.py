import numpy as np
import sys
import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve
from photutils.segmentation import detect_sources, make_2dgaussian_kernel, SourceCatalog
from photutils.isophote import EllipseGeometry, Ellipse, build_ellipse_model
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
import sys
from masking import obj_rej_mask
from utils import radec
from astropy.convolution import convolve

def norm(x, percent=99):
    return simple_norm(x, 'linear', percent=percent)

class EllipseIsophote:
    def __init__(self, path, obj_name):
        self.path = path
        self.obj = obj_name
    
    def data_imp(self):
        hdu = fits.getdata(self.path+'/sky_subed/coadd.fits')
        hdr = fits.getheader(self.path+'/sky_subed/coadd.fits')
        self.hdu = hdu
        self.hdr = hdr

    def obj_mask(self, threshold=float):
        ra, dec = radec(self.obj)
        mask, threshold, aper = obj_rej_mask(self.hdu, threshold, self.hdr, ra, dec, bkg_thrsh=True)
        self.mask = mask
        self.threshold = threshold
        self.aper = aper

    def exp_param(self):
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
        seg_map = detect_sources(conv_arr, np.nanstd(conv_arr), n_pixels=10)
        cat = SourceCatalog(conv_arr, seg_map)
        xcen,ycen = cat.x_centroid,cat.y_centroid
        idx = np.where((np.min(abs(x-xcen))==abs(x-xcen))&(np.min(abs(y-ycen)==abs(y-ycen))))
        cat1 = cat[idx][0]
        """
        pos = (cat1.x_centroid, cat1.y_centroid)
        aper = EllipticalAperture(pos, a=3*cat1.semimajor_axis.value, b=3*cat1.semiminor_axis.value, theta=cat1.orientation.value*np.pi/180)
        plt.imshow(seg_map, origin='lower')
        aper.plot(color='r', linewidth=0.7)
        plt.show();sys.exit()
        """
        h, pa, ell = cat1.semimajor_axis.value, cat1.orientation.value+90, cat1.ellipticity.value
        self.geo = EllipseGeometry(xcen[0], ycen[0], h, ell, (pa-90)*np.pi/180)
        self.h = h
        self.pa = pa
        self.ell = ell
        print(pa, ell, i0, h)
        return pa, ell, i0, h

    def sersic_param(self):
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
        seg_map = detect_sources(conv_arr, np.ma.std(conv_arr), n_pixels=2000)
        cat = SourceCatalog(conv_arr, seg_map)
        
        xcen,ycen = cat.x_centroid,cat.y_centroid
        idx = np.where((np.min(abs(x-xcen))==abs(x-xcen))&(np.min(abs(y-ycen)==abs(y-ycen))))
        cat1 = cat[idx][0]
        """
        pos = (cat1.x_centroid, cat1.y_centroid)
        aper = EllipticalAperture(pos, a=3*cat1.semimajor_axis.value, b=3*cat1.semiminor_axis.value, theta=cat1.orientation.value*np.pi/180)
        plt.imshow(seg_map, origin='lower')
        aper.plot(color='r', linewidth=0.7)
        plt.show()
        """
        r_e, pa, ell = cat1.semimajor_axis.value, cat1.orientation.value+90, cat1.ellipticity.value
        self.geo = EllipseGeometry(xcen[0], ycen[0], r_e, ell, (pa-90)*np.pi/180)
        self.h = r_e
        self.pa = pa
        self.ell = ell
        print(pa, ell, 4, i_e, r_e)
        return pa, ell, 4, i_e, r_e

    def ellipse(self):    
        img = np.ma.masked_array(self.hdu,self.mask)
        ellipse = Ellipse(img, geometry=self.geo)
        isolist = ellipse.fit_image(sma0=self.h, integrmode='bilinear',step=0.05, sclip=3.0, n_clip=3, fflag=0.3, fix_center=True) #isophote #maxsma=5*self.h,
        tbl = isolist.to_table()    
        print(tbl)
        model = build_ellipse_model(img.shape, isolist) #modeling
        return model, tbl

path = '~/NGC5907'
obj = 'NGC5907'
#hdu = fits.open(path+'/''.fits')[0].data
#mask = fits.open(path+'/obj_rejec_'+obj+'.fits')[0].data 

fig, ax = plt.subplots(1,3, figsize=(15,5), gridspec_kw={'width_ratios':[1,1,1]})
ellipse = EllipseIsophote(path, obj)
ellipse.data_imp()
ellipse.obj_mask(threshold=3)
ellipse.sersic_param()
model,tbl = ellipse.ellipse()
radius = tbl['sma'] * 1.86
intens = tbl['intens'] / (1.86**2)
mag = -2.5*np.log10(intens)+30.1
#tbl.write(path+'/test_color_iso_tbl_'+color+'.csv', format='ascii.csv', overwrite=True)
ax[0].scatter(radius, mag, s=2)
ax[1].imshow(np.log10(model), origin='lower')
ax[2].imshow(ellipse.hdu-model, norm=norm(ellipse.hdu, percent=90), origin='lower')
#sys.exit()
ax[0].set_xlabel('sma(arcsec)')
ax[0].set_ylabel('$\mu_r$')
ax[0].invert_yaxis()
plt.legend()
plt.show()