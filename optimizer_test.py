import numpy as np
import matplotlib.pyplot as plt
import sys
from cpnn import Variable
import cpnn.functions as F
from cpnn import Function
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip

path = '~/NGC5907'
file_name='coadd'
obj='NGC5907'
color='r'
data = Table.read(path+'/sky_subed/'+file_name+'.cat', format='ascii', converters={'obsid':str})
sdss = Table.read(path + '/sdss_'+obj+'.csv', format='ascii') #check!! 
sdsscat = sdss['ra', 'dec', 'g','r','u', 'Err_r', 'Err_g', 'Err_u']#[sdss[color]>15]
objcat = data['ALPHAPEAK_J2000','DELTAPEAK_J2000','FLUX_BEST','FLUXERR_BEST']#, 'MAGERR_BEST']#, 'ERRAWIN_IMAGE', 'ERRBWIN_IMAGE']
#obj_cat = objcat[(objcat['ERRAWIN_IMAGE']<0.01)&(objcat['ERRBWIN_IMAGE']<0.01)]
sdss_coord = SkyCoord(ra=sdsscat['ra'], dec=sdsscat['dec'],unit='deg', frame='fk5')
obj_coord = SkyCoord(ra=objcat['ALPHAPEAK_J2000'], dec=objcat['DELTAPEAK_J2000'],unit='deg', frame='fk5')

idx1, d2d1, d3d1 = sdss_coord.match_to_catalog_sky(obj_coord)
sdss_data = sdsscat
obj_f = objcat[idx1]

obj_flux = obj_f['FLUX_BEST']
sdss_mag1 = sdss_data[color]
z_m = -2.5*np.log10(np.array(obj_flux))
sdss_mag = np.array(sdss_mag1)


#m = -2.5*np.log10(count)
zm = sigma_clip(z_m, cenfunc='median', stdfunc='mad_std',sigma_lower=1.3,sigma_upper=1.8)
m = z_m[zm.mask==False]
mag = sdss_mag[zm.mask==False]
mM = mag - m

z =  sigma_clip(mM, cenfunc='median', stdfunc='mad_std',sigma=3)
pre_fluxerr = obj_f['FLUXERR_BEST'][zm.mask==False]
fluxerr = pre_fluxerr[z.mask==False]
t_r = m[z.mask==False]
z0 = np.ma.median(z)

x = Variable(t_r)
y = mag[z.mask==False]
#x0 = np.arange(-5,5,0.1)
#y0 = 1 / 1+np.exp(-x0)

W1 = Variable(np.array(0.5))
b1 = Variable(np.array(z0))

"""
W2 = Variable(0.01 * np.random.randn(H,O))
b2 = Variable(np.zeros(O))
"""
def predict(x):
    y = F.linear(x, W1, b1)
    return y

lr = 0.2
iters = 10000

class ChiSquaredError(Function):
    def __init__(self, sigma):
        self.sigma = sigma
    def forward(self, x0, x1):
        diff = ((x0 - x1)**2)/(self.sigma**2)
        y = diff.sum() #(diff**2).sum() /len(diff)
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2 / self.sigma**2)
        gx1 = -gx0
        return gx0, gx1
    
def chi_squared_error(x0, x1):
    return ChiSquaredError(fluxerr)(x0, x1)
    
for i in range(iters):
    y_pred = predict(x)
    loss = chi_squared_error(y, y_pred)#

    W1.cleargrad()
    b1.cleargrad()
    loss.backward()
    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    
    if i % 1000 == 0:
        print(loss)

x1 = Variable(np.linspace(-20,1,100).reshape(100,1))
y1 = predict(x1)
print(W1.data,b1.data)
plt.scatter(W1.data*x.data+b1.data, y.data-(W1.data*x.data+b1.data), s=3)
#plt.plot(x1.data, y1.data, c='red')
plt.xlabel('$L_{inst}$')
plt.ylabel('$\Delta mag$')
#plt.ylabel('SDSS-r')
plt.show()