from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np


import time


class WorkerSignals(QObject):


    emit_img = pyqtSignal(np.ndarray)
    finished=pyqtSignal()
    emit_converted_img=pyqtSignal(np.ndarray)
    emit_cam_status=pyqtSignal(np.ndarray)




class Acquisition_thread(QRunnable):

    def __init__(self, cam, *args, **kwargs):
        super(Acquisition_thread, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.signals=WorkerSignals()
        self.cam = cam
        #self.control_parameter= control_parameter


    @pyqtSlot()
    def run(self):
            '''print(self.control_parameter)
            self.cam.set_auto_exposure(self.control_parameter[0])
            self.cam.set_auto_gain(self.control_parameter[1])
            if not self.control_parameter[0]:
                self.cam.set_exp(self.control_parameter[2])
            if not self.control_parameter[1]:
                self.cam.set_exp(self.control_parameter[3])'''
            img = self.cam.read()
            #img = self.cam.to_numpy(img)
            img=img.GetNDArray()
            #img=img/255
            #print(img.dtype)
            #print(np.max(img))
            #print([self.cam.get_auto_gain(), self.cam.get_auto_exposure(), self.cam.get_gain(), self.cam.get_exp()])
            #print(self.cam.get_auto_exposure())
            result = np.array([int(self.cam.get_frame_rate()), int(self.cam.get_exp()), int(self.cam.get_gain())])
            self.signals.emit_cam_status.emit(result)
            self.signals.emit_img.emit(img)






class Image_retrieval(QRunnable):

    def __init__(self, Phaseimg, mode,*args, **kwargs):
        super(Image_retrieval, self).__init__()

        self.args = args
        self.kwargs = kwargs
        self.Phaseimg=Phaseimg
        self.signals = WorkerSignals()
        self.mode=mode

    @pyqtSlot()
    def run(self):
        try:
            if self.mode==0:
                phase= self.Phaseimg.calculate_phase()
                self.signals.emit_converted_img.emit(phase)
            if self.mode==1:
                height= self.Phaseimg.calculate_height()
                self.signals.emit_converted_img.emit(height)
        finally:
            self.signals.finished.emit()





