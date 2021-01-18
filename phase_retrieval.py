import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import math
import numpy as np
import tifffile
from PIL import Image
import scipy.io
from skimage.restoration import unwrap_phase
import math

radi=200
#raw_image=cv2.imread("C:/Users/lambc/Dropbox/Deep-learning_PhaseRetrieval/1230-spacer-sample.bmp",0)
#cal_image=cv2.imread("background.tif",0)
raw_image=cv2.imread("C:/Users/lambc/Dropbox/project/vacuole/pollen/sample7.bmp",0)
cal_image=cv2.imread("C:/Users/lambc/Dropbox/project/vacuole/pollen/bg7.bmp", 0)
#plt.imshow(raw_image)
#plt.show()
fourier_raw_image= np.fft.fftshift(np.fft.fft2(raw_image))
plt.imshow(np.log(np.abs(fourier_raw_image)))
plt.show()
half_raw_fimage=fourier_raw_image[np.int(0.6*fourier_raw_image.shape[0]):,:]
#half_raw_fimage=fourier_raw_image[:,np.int(0.6*fourier_raw_image.shape[1]):]
half_raw_fimage=np.log((np.abs(half_raw_fimage)))
### find the +1 point
ind1=np.unravel_index(np.argmax(half_raw_fimage,axis=None),half_raw_fimage.shape)
orig_ind1=[ind1[0]+np.int(0.6*fourier_raw_image.shape[0]),ind1[1]]
#orig_ind1=[ind1[0],ind1[1]+np.int(0.6*fourier_raw_image.shape[1])]
### find the corresponding circle area
x_array=range(raw_image.shape[0])
y_array=range(raw_image.shape[1])
x_sub=x_array-orig_ind1[0]
y_sub=y_array-orig_ind1[1]
x_2=x_sub*x_sub
y_2=y_sub*y_sub
x_mat=np.tile(x_2,(raw_image.shape[1],1))
y_mat=np.tile(y_2,(raw_image.shape[0],1))
x_mat=np.transpose(x_mat)
polar_mat=x_mat+y_mat
chose_area=np.where(polar_mat<radi*radi,1,0)
filter_raw_image=(fourier_raw_image*chose_area)
plt.imshow(np.log(np.abs(filter_raw_image+1)))
plt.show()
### shift to the center
filter_raw_image=np.roll(filter_raw_image,-(orig_ind1[1]-np.int(0.5*fourier_raw_image.shape[1])),axis=1)
filter_raw_image=np.roll(filter_raw_image,-(orig_ind1[0]-np.int(0.5*fourier_raw_image.shape[0])),axis=0)
#plt.imshow(np.log(np.abs(filter_raw_image+1)))
#plt.show()
filter_raw_image=np.fft.ifft2(np.fft.ifftshift(filter_raw_image))
#plt.imshow(np.arctan2(np.imag(filter_raw_image),np.real(filter_raw_image)))
#plt.colorbar()
#plt.show()
### processing calibration image
fourier_cal_image=np.fft.fftshift(np.fft.fft2(cal_image))
filter_cal_image=fourier_cal_image*chose_area
filter_cal_image=np.roll(filter_cal_image,-(orig_ind1[1]-np.int(0.5*fourier_raw_image.shape[1])),axis=1)
filter_cal_image=np.roll(filter_cal_image,-(orig_ind1[0]-np.int(0.5*fourier_raw_image.shape[0])),axis=0)
#plt.imshow(np.log(np.abs(filter_cal_image+1)))
#plt.show()
filter_cal_image=np.fft.ifft2(np.fft.ifftshift(filter_cal_image))
#plt.imshow(np.arctan2(np.imag(filter_cal_image),np.real(filter_cal_image)))
#plt.colorbar()
#plt.show()
result_image=np.angle(filter_raw_image/filter_cal_image)
plt.imshow(result_image)
plt.colorbar()
plt.show()
unwrap_result_image=unwrap_phase(result_image)
x_conf=-1*np.mean(unwrap_result_image[:,30]-unwrap_result_image[:,-30])
y_conf=-1*np.mean(unwrap_result_image[30,:]-unwrap_result_image[-30,:])
x_conf_vec=np.linspace(x_conf,0,unwrap_result_image.shape[1])
y_conf_vec=np.linspace(y_conf,0,unwrap_result_image.shape[0])
unwrap_result_image=unwrap_result_image+x_conf_vec
unwrap_result_image=unwrap_result_image+y_conf_vec.reshape(unwrap_result_image.shape[0],1)
unwrap_result_image=unwrap_result_image-np.mean(unwrap_result_image[0:100,0:100])
lambd=0.532
dn=0.04
height_image=unwrap_result_image*lambd/2/math.pi/dn
scipy.io.savemat('height_fast_unwrap.mat', mdict={'arr':height_image})
np.save('phase_0317', unwrap_result_image)
plt.imshow(-unwrap_result_image,cmap=plt.cm.jet)
plt.clim(-3,7)
plt.colorbar()
plt.show()
