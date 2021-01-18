import numpy as np
from skimage.restoration import unwrap_phase
import scipy
import math

class PhaseImg(object):

    def __init__(self):
        self.raw_image=np.ndarray([])
        self.cal_image=np.ndarray([])
        self.radi=120
        self.lamda=0.532
        self.dn=0.04
        self.colorbar_min=0
        self.colorbar_max=7
        self.phase_image = np.ndarray([])
        self.height_image = np.ndarray([])
        self.width=0
        self.height=0
        self.zoom_index=4.8/44.4

    def set_filter_size(self, filter_size):
        self.filter_size=filter_size


    def set_wavelength(self, wavelength):
        self.lamda=wavelength


    def set_dn(self,dn):
        self.dn=dn

    def set_raw_image(self,raw_image):
        self.raw_image=raw_image


    def set_cal_image(self,cal_image):
        self.cal_image=cal_image
        self.width=cal_image.shape[1]
        self.height=cal_image.shape[0]

    def calculate_phase(self):
        # plt.imshow(raw_image)
        # plt.show()
        fourier_raw_image = np.fft.fftshift(np.fft.fft2(self.raw_image))
        # plt.imshow(np.log(np.abs(fourier_raw_image)))
        # plt.show()
        half_raw_fimage = fourier_raw_image[np.int(0.6 * fourier_raw_image.shape[0]):, :]
        #half_raw_fimage=fourier_raw_image[:,np.int(0.6*fourier_raw_image.shape[1]):]
        half_raw_fimage = np.log((np.abs(half_raw_fimage)))
        ### find the +1 point
        ind1 = np.unravel_index(np.argmax(half_raw_fimage, axis=None), half_raw_fimage.shape)
        orig_ind1 = [ind1[0] + np.int(0.6 * fourier_raw_image.shape[0]), ind1[1]]
        #orig_ind1=[ind1[0],ind1[1]+np.int(0.6*fourier_raw_image.shape[1])]
        ### find the corresponding circle area
        x_array = range(self.raw_image.shape[0])
        y_array = range(self.raw_image.shape[1])
        x_sub = x_array - orig_ind1[0]
        y_sub = y_array - orig_ind1[1]
        x_2 = x_sub * x_sub
        y_2 = y_sub * y_sub
        x_mat = np.tile(x_2, (self.raw_image.shape[1], 1))
        y_mat = np.tile(y_2, (self.raw_image.shape[0], 1))
        x_mat = np.transpose(x_mat)
        polar_mat = x_mat + y_mat
        chose_area = np.where(polar_mat < self.radi * self.radi, 1, 0)
        filter_raw_image = (fourier_raw_image * chose_area)
        #plt.imshow(np.log(np.abs(filter_raw_image + 1)))
        #plt.show()
        ### shift to the center
        filter_raw_image = np.roll(filter_raw_image, -(orig_ind1[1] - np.int(0.5 * fourier_raw_image.shape[1])), axis=1)
        filter_raw_image = np.roll(filter_raw_image, -(orig_ind1[0] - np.int(0.5 * fourier_raw_image.shape[0])), axis=0)
        # plt.imshow(np.log(np.abs(filter_raw_image+1)))
        # plt.show()
        filter_raw_image = np.fft.ifft2(np.fft.ifftshift(filter_raw_image))
        # plt.imshow(np.arctan2(np.imag(filter_raw_image),np.real(filter_raw_image)))
        # plt.colorbar()
        # plt.show()
        ### processing calibration image
        fourier_cal_image = np.fft.fftshift(np.fft.fft2(self.cal_image))
        filter_cal_image = fourier_cal_image * chose_area
        filter_cal_image = np.roll(filter_cal_image, -(orig_ind1[1] - np.int(0.5 * fourier_raw_image.shape[1])), axis=1)
        filter_cal_image = np.roll(filter_cal_image, -(orig_ind1[0] - np.int(0.5 * fourier_raw_image.shape[0])), axis=0)
        # plt.imshow(np.log(np.abs(filter_cal_image+1)))
        # plt.show()
        filter_cal_image = np.fft.ifft2(np.fft.ifftshift(filter_cal_image))
        # plt.imshow(np.arctan2(np.imag(filter_cal_image),np.real(filter_cal_image)))
        # plt.colorbar()
        # plt.show()
        result_image = np.angle(filter_raw_image / filter_cal_image)
        # plt.imshow(result_image)
        # plt.colorbar()
        # plt.show()
        unwrap_result_image = unwrap_phase(result_image)
        x_conf=-1*np.mean(unwrap_result_image[:,30]-unwrap_result_image[:,-30])
        y_conf=-1*np.mean(unwrap_result_image[30,:]-unwrap_result_image[-30,:])
        x_conf_vec=np.linspace(x_conf,0,unwrap_result_image.shape[1])
        y_conf_vec=np.linspace(y_conf,0,unwrap_result_image.shape[0])
        unwrap_result_image=unwrap_result_image+x_conf_vec
        unwrap_result_image=unwrap_result_image+y_conf_vec.reshape(unwrap_result_image.shape[0],1)
        unwrap_result_image = unwrap_result_image - np.mean(unwrap_result_image[0:100, 0:100])
        unwrap_result_image = scipy.ndimage.median_filter(unwrap_result_image, size=3)
        self.phase_image=unwrap_result_image
        self.height_image= self.phase_image * self.lamda / 2 / math.pi / self.dn
        return unwrap_result_image

    def calculate_height(self):
        self.calculate_phase()
        return self.height_image


