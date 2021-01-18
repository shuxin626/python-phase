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
from skimage.segmentation import watershed, clear_border
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

def phase_retrieval(raw_image_path, cal_image_path, mode=0, radi=70, wavelength=0.532, dn=0.04):
## raw_image_path: path of image containing sample
## cal_image_path: path of backgroud image
## mode : 0 outputs phase_image, 1 outputs height_image
## radi: size of bandpass filter
## lambda: wavelenth of light source, needed when output height_image
## dn: refractive contrast between buffer and sample, needed when output phase_image

    raw_image=cv2.imread(raw_image_path,0)
    cal_image=cv2.imread(cal_image_path,0)
    #plt.imshow(raw_image)
    #plt.show()
    fourier_raw_image= np.fft.fftshift(np.fft.fft2(raw_image))
    #plt.imshow(np.log(np.abs(fourier_raw_image)))
    #plt.show()
    #half_raw_fimage=fourier_raw_image[np.int(0.6*fourier_raw_image.shape[0]):,:] ## denpends on the direction of fringe
    half_raw_fimage=fourier_raw_image[:,np.int(0.6*fourier_raw_image.shape[1]):]
    half_raw_fimage=np.log((np.abs(half_raw_fimage)))
    ### find the +1 point
    ind1=np.unravel_index(np.argmax(half_raw_fimage,axis=None),half_raw_fimage.shape)
    #orig_ind1=[ind1[0]+np.int(0.6*fourier_raw_image.shape[0]),ind1[1]]
    orig_ind1=[ind1[0],ind1[1]+np.int(0.6*fourier_raw_image.shape[1])]
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
    #plt.imshow(np.log(np.abs(filter_raw_image+1)))
    #plt.show()
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
    #plt.imshow(result_image)
    #plt.colorbar()
    #plt.show()
    unwrap_result_image=unwrap_phase(result_image)
    x_conf=-1*np.mean(unwrap_result_image[:,30]-unwrap_result_image[:,-30])
    y_conf=-1*np.mean(unwrap_result_image[30,:]-unwrap_result_image[-30,:])
    x_conf_vec=np.linspace(x_conf,0,unwrap_result_image.shape[1])
    y_conf_vec=np.linspace(y_conf,0,unwrap_result_image.shape[0])
    unwrap_result_image=unwrap_result_image+x_conf_vec
    unwrap_result_image=unwrap_result_image+y_conf_vec.reshape(unwrap_result_image.shape[0],1)
    unwrap_result_image=unwrap_result_image-np.mean(unwrap_result_image[0:100,0:100])
    #lambd=0.532
    #dn=0.04
    if mode==0:
        return(unwrap_result_image)
    else:
        height_image=unwrap_result_image*wavelength/2/math.pi/dn
        return(height_image)
    #scipy.io.savemat('height_fast_unwrap.mat', mdict={'arr':height_image})
    #plt.imshow(unwrap_result_image,cmap=plt.cm.jet)
    #np.save('phase_0317', unwrap_result_image)
    #plt.clim(-0.5,1)
    #plt.colorbar()
    #plt.show()


def segmentation(src, threshold):
## Apated from https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
## https://github.com/RohanTrix/Osteosarcoma-cell-Segmentation-using-Watershed/blob/master/Cell_segment.py
## This watershed algorithm accept single channel input src
## Threshold is used to segment cell from background, threshold=[min,max]
## Output segmentation label mask with background as 0, cell1 as 1, cell2 as 2 ...
    if src is None:
        #print('Could not open or find the image:', args.input)
        print('Could not open or find the image')
        exit(0)
    src=np.clip(src, threshold[0], threshold[1]) # threshold lower bound is to remove backgrond noise, upper bound is to remove outliers.
    cv2.normalize(src, src, 0, 255, cv2.NORM_MINMAX)
    src=src.astype('uint8')
    plt.imshow(src,cmap=plt.cm.jet)
    plt.colorbar()
    plt.show()
    # Threshold
    _, bw = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #plt.imshow(bw)
    #plt.show()
    kernel1 = np.ones((10,10),np.uint8)
    closing = cv2.morphologyEx(bw,cv2.MORPH_CLOSE, kernel1, iterations=5) ## Remove small points
    closing= clear_border(closing)
    erosion=cv2.erode(closing, kernel1, iterations=2) 
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(erosion)
    norm_dist=np.zeros(distance.shape)
    cv2.normalize(distance, norm_dist, 0, 1, cv2.NORM_MINMAX)
    _, mask = cv2.threshold(norm_dist, 0.7, 1.0, cv2.THRESH_BINARY)
    #coords = peak_local_max(distance, footprint=np.ones((100, 100)), labels=bw)
    #mask = np.zeros(distance.shape, dtype=bool)
    #mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=bw)
    fig, axes = plt.subplots(ncols=4, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(bw, cmap=plt.cm.gray)
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(erosion, cmap=plt.cm.gray)
    ax[1].set_title('Erosion')  
    ax[2].imshow(-distance, cmap=plt.cm.gray)
    ax[2].set_title('Distances')
    ax[3].imshow(labels, cmap=plt.cm.nipy_spectral)
    ax[3].set_title('Separated objects')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()
    return labels
    ## it is better to add an active countour algorithm, see scikit-image