from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from dnn import myUnet
import global_variables
import math

import numpy as np
import tensorflow as tf
from keras import backend as K
import os 
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


class WorkerSignals(QObject):


    emit_img = pyqtSignal(np.ndarray)
    #finished=pyqtSignal()
    emit_converted_img=pyqtSignal(np.ndarray)
    emit_cam_status=pyqtSignal(np.ndarray)




class Acquisition_thread(QRunnable):

    def __init__(self, cam, *args, **kwargs):
        super(Acquisition_thread, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.signals=WorkerSignals()
        self.cam = cam
        self._mutex=QMutex()
        #self.control_parameter= control_parameter
        self.keep_aquisition=True


    @pyqtSlot()
    def run(self):
        while self.keep_aquisition==True:
            '''print(self.control_parameter)
            self.cam.set_auto_exposure(self.control_parameter[0])
            self.cam.set_auto_gain(self.control_parameter[1])
            if not self.control_parameter[0]:
                self.cam.set_exp(self.control_parameter[2])
            if not self.control_parameter[1]:
                self.cam.set_exp(self.control_parameter[3])'''
            img = self.cam.read()
            #img = self.cam.to_numpy(img)
            global_variables.mutex1.lock()
            global_variables.raw_image=img.GetNDArray()
            global_variables.mutex1.unlock()
            #img=img/255
            #print(img.dtype)
            #print(np.max(img))
            #print([self.cam.get_auto_gain(), self.cam.get_auto_exposure(), self.cam.get_gain(), self.cam.get_exp()])
            #print(self.cam.get_auto_exposure())
            result = np.array([int(self.cam.get_frame_rate()), int(self.cam.get_exp()), int(self.cam.get_gain())])
            self.signals.emit_cam_status.emit(result)
            self.signals.emit_img.emit(img.GetNDArray())
            time.sleep(0.007)

    def stop(self):
        self._mutex.lock()
        self.keep_aquisition=False
        self._mutex.unlock()







class Image_retrieval(QRunnable):

    def __init__(self, mode,*args, **kwargs):
        super(Image_retrieval, self).__init__()

        self.args = args
        self.kwargs = kwargs
        #self.Phaseimg=Phaseimg
        self.signals = WorkerSignals()
        self.mode=mode
        self._mutex=QMutex()
        self.keep_converting=True



    @pyqtSlot()
    def run(self):
        self.network_init()
        while self.keep_converting==True:
            global_variables.mutex1.lock()
            image=img_to_array(global_variables.raw_image)
            global_variables.mutex1.unlock()
            image = np.expand_dims(image, axis=0)
            #print(image.shape)
            image /= 127.5 - 1.0 #norm
            predict_array=self.model.predict(image, batch_size=1, verbose=1)
            phase=np.squeeze(predict_array, axis=0)
            phase=phase*12
            #phase=array_to_img(predict_array)
            if self.mode==0:
                #phase= self.Phaseimg.calculate_phase()
                self.signals.emit_converted_img.emit(phase)
            if self.mode==1:
                height = phase * 0.532 / 2 / math.pi / 0.04 
                self.signals.emit_converted_img.emit(height)
            time.sleep(0.003)

    def network_init(self):
        K.clear_session()
        myunet=myUnet()
        self.model=myunet.UNet()
        self.model.load_weights('T3T4Patch2712_Net2MSE2_3t3_U2_69layers.hdf5')
        #print('loading network')
        self.model._make_predict_function()

    def stop(self):
        self._mutex.lock()
        self.keep_converting=False
        self._mutex.unlock()

    def change_mode(self, mode):
        self._mutex.lock()
        self.mode=mode
        self._mutex.unlock()




