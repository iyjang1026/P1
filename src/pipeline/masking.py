import numpy as np
from astropy.convolution import convolve
from photutils.segmentation import detect_sources, make_2dgaussian_kernel, SourceCatalog, deblend_sources, SegmentationImage
from photutils.background import MedianBackground, Background2D
from photutils.aperture import EllipticalAperture
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

def simple_masking(arr):
    bkg_estimator = MedianBackground()
    bkg = Background2D(arr, (64,64), filter_size=(3,3), bkg_estimator=bkg_estimator)
    data = arr - bkg.background
    threshold = 1.5 * bkg.background_rms
    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolve(data, kernel)
    seg_map = detect_sources(convolved_data, threshold, npixels=30)
    mask_map = np.array(seg_map)
    kernel = disk(10)
    mask_map_d = binary_dilation(mask_map, kernel, iterations=3)
    masked = np.where((mask_map_d!=0), 1, 0)
    return masked.astype(np.int8)

def region_mask(hdu, thrsh,pix_scale,disk_r=100,ampglow=True):
    z_arr = np.ma.masked_equal(hdu, 0)
    if type(ampglow)==bool:
        if ampglow==True:
            half = disk(disk_r)
            z_arr = np.zeros_like(hdu)
            z_arr[2048-disk_r:2048,1212-disk_r-1:1212+disk_r] += half[0:disk_r,:]
        elif ampglow==False:
            half = None
            z_arr = np.where(hdu!=0, False, True)
    elif type(ampglow) == np.ndarray:
        half = disk(disk_r)
        z_arr0 = np.array(ampglow) #np.where(ampglow!=0,False,True)
        z_arr0[2048-disk_r:2048,1212-disk_r-1:1212+disk_r] += half[0:disk_r,:]
        z_arr = np.where(z_arr0!=0,1,0)

    bkg_est = MedianBackground()
    bkg = Background2D(hdu, (64,64), filter_size=(5,5), bkg_estimator=bkg_est, mask=z_arr)
    data = hdu - bkg.background
    threshold = thrsh*bkg.background_rms
    kernel = make_2dgaussian_kernel(fwhm=3/pix_scale, size=5)
    conv_hdu = convolve(data, kernel)
    seg_map = detect_sources(conv_hdu, threshold, npixels=5, mask=z_arr)
    segm_deblend = deblend_sources(conv_hdu, seg_map,
                               npixels=2000,connectivity=8, mode='exponential', nlevels=32, contrast=0.001,
                               progress_bar=False)
    seg = np.array(seg_map)
    
    cat = SourceCatalog(data, segm_deblend, convolved_data=conv_hdu)

    a_list = list(cat.semiminor_sigma.value)
    arr_zero = np.zeros_like(hdu).astype(np.float32) 
    tmp = a_list.copy()
    tmp.sort()
    tmp_num = tmp[-20:]
    top_idx = [a_list.index(x) for x in tmp_num]
    for i in top_idx:
        """
        g_aper = l[i]
        a = g_aper.a
        b = g_aper.b
        xypos = g_aper.positions
        theta = g_aper.theta
        xy = (int(xypos[0]), int(xypos[1]))
        """
        cat0 = cat[i]
        xy = (cat0.xcentroid, cat0.ycentroid)
        theta = cat0.orientation.value *np.pi /180
        a,b = 3*cat0.semimajor_sigma.value, 3*cat0.semiminor_sigma.value
        #aperture = EllipticalAperture(xy, 3*a, 3*b, theta)
        aperture = EllipticalAperture(xy, 3.5*a, 3.5*b, theta=theta)
        mask = np.array(aperture.to_mask(method='center')).astype(np.int8)
        mask_x, mask_y = mask.shape
    
        st_x = np.int16(xy[1] - mask_x/2)
        st_y = np.int16(xy[0] - mask_y/2)
    
        x, y = hdu.shape
   
        def lim(st, mask_s, arr_s):
            if st < 0 and st+mask_s<arr_s:
                arr_st = 0
                mask_st = -st
                mask_l = mask_s
            elif st<0 and st+mask_s>arr_s:
                arr_st = 0
                mask_st = -st
                mask_l = mask_s + st - arr_s
        
            elif st+mask_s > arr_s:
                arr_st = st
                mask_st = 0
                mask_l = arr_s - st

            else:
                arr_st = st
                mask_st = 0
                mask_l = mask_s
            return arr_st, mask_st, mask_l
        
        arr_x, mask_s_x, mask_l_x = lim(st_x, mask_x, x)
        arr_y, mask_s_y, mask_l_y = lim(st_y, mask_y, y)
        mask = mask[mask_s_x:mask_l_x,mask_s_y:mask_l_y] 
        m_x, m_y = mask.shape #crop mask
        arr_zero[arr_x:arr_x+m_x, arr_y:arr_y+m_y] += mask
    
    kernel0 = disk(3) 
    seg_d= binary_dilation(seg, kernel0, iterations=1)
    masked_map = np.where(seg_d!=0, 1, 0) + arr_zero
    if type(ampglow)!=np.ndarray:
        if ampglow == True:
            masked_map[2048-disk_r:2048,1212-disk_r-1:1212+disk_r] += half[0:disk_r,:]
    elif type(half)==np.ndarray:
        if type(ampglow)==np.ndarray:
            masked_map += z_arr
    
    masked = np.where(masked_map!=0, 1, 0).astype(np.int8)
    return np.array(masked, dtype=np.int8)

def obj_rej_mask(hdu, thrsh,hdr,ra,dec):
    mask1 = np.where(hdu!=0, False, True)
    bkg_est = MedianBackground()
    bkg = Background2D(hdu, (64,64), filter_size=(3,3), bkg_estimator=bkg_est, mask=mask1)
    data = hdu - bkg.background
    threshold = thrsh*bkg.background_rms
    kernel = make_2dgaussian_kernel(fwhm=3.0, size=5)
    conv_hdu = convolve(data, kernel)
    seg_map = detect_sources(conv_hdu, threshold, npixels=9, mask=mask1) #1차 천체 탐지
    segm_deblend = deblend_sources(conv_hdu, seg_map,
                               npixels=2000, nlevels=32, contrast=0.0005,
                               progress_bar=False) #천체분리

    segm_d = np.array(segm_deblend).astype(np.int32)

    cat = SourceCatalog(hdu,segm_deblend, convolved_data=conv_hdu)

    a_list = list(cat.semiminor_sigma.value)
    
    arr_zero = np.zeros_like(hdu).astype(np.float32) 
    tmp = a_list.copy()
    tmp.sort()
    tmp_num = tmp[-20:]
    top_idx = [a_list.index(x) for x in tmp_num]
    cat = cat[top_idx]
    x,y = cat.xcentroid,cat.ycentroid
    w = WCS(hdr)
    eq_coord = SkyCoord(ra,dec,frame='fk5',unit='deg')
    pix_x,pix_y = w.world_to_pixel(eq_coord)
    idx = np.where((np.min(abs(pix_x-x))==abs(pix_x-x))&(np.min(abs(pix_y-y)==abs(pix_y-y))))
    cat1 = cat[idx]
 
    segm_d[cat1.bbox_ymin[0]:cat1.bbox_ymax[0],cat1.bbox_xmin[0]:cat1.bbox_xmax[0]] = 0
    #plt.imshow(segm_d,origin='lower');plt.show();sys.exit()
    segm = SegmentationImage(segm_d)
    cat = SourceCatalog(segm, segm, convolved_data=conv_hdu)
     
    a_list = list(cat.semiminor_sigma.value)
    
    arr_zero = np.zeros_like(hdu).astype(np.float32) 
    tmp = a_list.copy()
    tmp.sort()
    tmp_num = tmp[-3:]
    top_idx = [a_list.index(x) for x in tmp_num]
    for i in top_idx:
        """
        g_aper = l[i]
        a = g_aper.a
        b = g_aper.b
        xypos = g_aper.positions
        theta = g_aper.theta
        xy = (int(xypos[0]), int(xypos[1]))
        """
        cat0 = cat[i]
        xy = (cat0.xcentroid, cat0.ycentroid)
        theta = cat0.orientation.value *np.pi /180
        a,b = 3*cat0.semimajor_sigma.value, 3*cat0.semiminor_sigma.value
        aperture = EllipticalAperture(xy, 2.5*a, 2.5*b, theta)
        mask = np.array(aperture.to_mask(method='center')).astype(np.int8)
        mask_x, mask_y = mask.shape
    
        st_x = np.int16(xy[1] - mask_x/2)
        st_y = np.int16(xy[0] - mask_y/2)
    
        x, y = hdu.shape
   
        def lim(st, mask_s, arr_s):
            if st < 0 and st+mask_s<arr_s:
                arr_st = 0
                mask_st = -st
                mask_l = mask_s
            elif st<0 and st+mask_s>arr_s:
                arr_st = 0
                mask_st = -st
                mask_l = mask_s + st - arr_s
        
            elif st+mask_s > arr_s:
                arr_st = st
                mask_st = 0
                mask_l = arr_s - st

            else:
                arr_st = st
                mask_st = 0
                mask_l = mask_s
            return arr_st, mask_st, mask_l
        
        arr_x, mask_s_x, mask_l_x = lim(st_x, mask_x, x)
        arr_y, mask_s_y, mask_l_y = lim(st_y, mask_y, y)
        mask = mask[mask_s_x:mask_l_x,mask_s_y:mask_l_y] 
        m_x, m_y = mask.shape #crop mask
        arr_zero[arr_x:arr_x+m_x, arr_y:arr_y+m_y] += mask
    kernel0 = disk(3) 
    segm_d = binary_dilation(segm_d, kernel0, iterations=1) #ngrow
    masked_map = np.where(segm_d!=0, 1, 0) + arr_zero #region 마스크 영상과 segmentation 마스크 영상을 합침
    masked = np.where(masked_map!=0, 1, 0).astype(np.int8)
    
    return np.array(masked, dtype=np.int8)

def psf_obj_rej_mask(hdu, thrsh,hdr,ra,dec):
    mask1 = np.where(hdu!=0, False, True)
    bkg_est = MedianBackground()
    bkg = Background2D(hdu, (64,64), filter_size=(3,3), bkg_estimator=bkg_est, mask=mask1)
    data = hdu - bkg.background
    threshold = thrsh*bkg.background_rms
    kernel = make_2dgaussian_kernel(fwhm=3.0, size=5)
    conv_hdu = convolve(data, kernel)
    seg_map = detect_sources(conv_hdu, threshold, npixels=9, mask=mask1) #1차 천체 탐지
    segm_deblend = deblend_sources(conv_hdu, seg_map,
                               npixels=2000, nlevels=32, contrast=0.0005,
                               progress_bar=False) #천체분리

    segm_d = np.array(segm_deblend).astype(np.int32)

    cat = SourceCatalog(hdu,segm_deblend, convolved_data=conv_hdu)

    a_list = list(cat.semiminor_sigma.value)
    
    arr_zero = np.zeros_like(hdu).astype(np.float32) 
    tmp = a_list.copy()
    tmp.sort()
    tmp_num = tmp[-20:]
    top_idx = [a_list.index(x) for x in tmp_num]
    cat = cat[top_idx]
    x,y = cat.xcentroid,cat.ycentroid
    w = WCS(hdr)
    eq_coord = SkyCoord(ra,dec,frame='fk5',unit='deg')
    pix_x,pix_y = w.world_to_pixel(eq_coord)
    idx = np.where((np.min(abs(pix_x-x))==abs(pix_x-x))&(np.min(abs(pix_y-y)==abs(pix_y-y))))
    
    arr_zero = np.zeros_like(hdu).astype(np.float32) 
    for i in idx:
        """
        g_aper = l[i]
        a = g_aper.a
        b = g_aper.b
        xypos = g_aper.positions
        theta = g_aper.theta
        xy = (int(xypos[0]), int(xypos[1]))
        """
        cat0 = cat[i][0]
        xy = (cat0.xcentroid, cat0.ycentroid)
        theta = cat0.orientation.value *np.pi /180
        a,b = 3*cat0.semimajor_sigma.value, 3*cat0.semiminor_sigma.value
        aperture = EllipticalAperture(xy, a, b, theta)
        mask = np.array(aperture.to_mask(method='center')).astype(np.int8)
        mask_x, mask_y = mask.shape
    
        st_x = np.int16(xy[1] - mask_x/2)
        st_y = np.int16(xy[0] - mask_y/2)
    
        x, y = hdu.shape
   
        def lim(st, mask_s, arr_s):
            if st < 0 and st+mask_s<arr_s:
                arr_st = 0
                mask_st = -st
                mask_l = mask_s
            elif st<0 and st+mask_s>arr_s:
                arr_st = 0
                mask_st = -st
                mask_l = mask_s + st - arr_s
        
            elif st+mask_s > arr_s:
                arr_st = st
                mask_st = 0
                mask_l = arr_s - st

            else:
                arr_st = st
                mask_st = 0
                mask_l = mask_s
            return arr_st, mask_st, mask_l
        
        arr_x, mask_s_x, mask_l_x = lim(st_x, mask_x, x)
        arr_y, mask_s_y, mask_l_y = lim(st_y, mask_y, y)
        mask = mask[mask_s_x:mask_l_x,mask_s_y:mask_l_y] 
        m_x, m_y = mask.shape #crop mask
        arr_zero[arr_x:arr_x+m_x, arr_y:arr_y+m_y] += mask
    kernel0 = disk(3) 
    segm_d = binary_dilation(segm_d, kernel0, iterations=1) #ngrow
    masked_map = arr_zero #region 마스크 영상과 segmentation 마스크 영상을 합침
    masked = np.where(masked_map!=0, 1, 0).astype(np.int8)
    
    return np.array(masked, dtype=np.int8)