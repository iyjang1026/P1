from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting
from scipy.interpolate import Rbf
import numpy as np
import warnings
import sys

warnings.filterwarnings('ignore')
    
def poly_sky_model(data, bin, order=2):
    img_height, img_width = data.shape

    newImage = np.zeros((bin,bin), dtype=data.dtype)

    new_height = img_height//bin
    new_width = img_width//bin

    """
    the center position of binned pixel
    """

    h_width = img_height//bin
    w_width = img_width//bin
    x0 = np.arange(0,img_width,w_width)
    y0 = np.arange(0,img_height,h_width)
    x,y = np.meshgrid(x0+w_width//2,y0+h_width//2, indexing='ij')   

    """
    binning
    """
    for j in range(bin):
        for i in range(bin):
            y1 = j*new_height
            x1 = i*new_width
            pixel = data[y1:y1+new_height, x1:x1+new_width]
            mean, median,std = sigma_clipped_stats(pixel, cenfunc='median',stdfunc='mad_std',sigma=3)
            newImage[j,i] = median
            
    """
    calculate matrix x and y, these are positon component or img
    """        
  
    data_nc = np.ma.masked_invalid(newImage)

    """
    modeling
    """

    p_init = models.Polynomial2D(degree=order) #다항함수 모델링
    fit_p = fitting.LinearLSQFitter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = fit_p(p_init, x, y, data_nc) #하늘의 모델을 반환(x,y)
    x_model,y_model = np.meshgrid(np.arange(img_width),np.arange(img_height), indexing='ij')
    return model(x_model, y_model)

def rbf_sky_model(data, bin):
    img_height, img_width = data.shape

    newImage = np.zeros((bin,bin), dtype=np.float16)

    new_height = img_height//bin
    new_width = img_width//bin

    """
    the center position of binned pixel
    """
    xx_m = np.arange(0,img_width, img_width/bin) + new_width//2
    yy_m = np.arange(0, img_height, img_height/bin) + new_height//2
    
    x_grid, y_grid = np.meshgrid(xx_m,yy_m, indexing='ij')
    
    """
    binning
    """
    for j in range(bin):
        for i in range(bin):
            y = j*new_height
            x = i*new_width
            pixel = data[y:y+new_height, x:x+new_width]
            mean, median,std = sigma_clipped_stats(pixel, cenfunc='median',stdfunc='mad_std',sigma=3)
            newImage[j,i] += np.float16(median)
            
    """
    calculate matrix x and y, these are positon component or img
    """
    x1,y1 = np.meshgrid(np.arange(img_width),np.arange(img_width))
    mask = np.isnan(newImage).astype(np.int8)
    data_nc = np.ma.masked_array(newImage,mask).astype(np.float16)
    x = np.ma.masked_array(x_grid, mask)
    y = np.ma.masked_array(y_grid, mask)

    """
    modeling
    """
    model = Rbf(y,x, data_nc, function='linear')#RBFInterpolator(coord,data_nc.ravel(),kernel='linear')
    return model(x1,y1)

