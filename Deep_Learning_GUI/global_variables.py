from PyQt5 import QtCore
import numpy as np


global mutex1
mutex1 = QtCore.QMutex()
global raw_image
raw_image=np.zeros([512,512], dtype='uint8')
global dn
dn=0.04
global wavelength
wavelength=0.532
global mag
mag=44.4
global pixel
pixel=4.8