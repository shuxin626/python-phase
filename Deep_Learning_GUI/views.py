from PhaseImage import PhaseImg
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from pyqtgraph import *
from multi_thread import *
import numpy as np
import PyQt5.QtCore
from model import FLIRCamDev
from Tab import Analyzer_Window
from matplotlib import cm
import global_variables


#retrieved_image=np.zeros([512,512], dtype='uint8')


class StartWindow(QMainWindow):
    def __init__(self, cam):
        super().__init__()

        self.central_widget = QWidget()
        self.tabWidget = QTabWidget()
        self.tabWidget.setTabsClosable(True)
        ##
        self.label_frame_rate = QLabel('Frame Rate')
        self.label_frame_rate_value = QLabel()
        self.label_exp_auto = QLabel('Exposure Auto')
        self.check_exp_auto = QCheckBox()
        self.check_exp_auto.setTristate(False)
        self.label_exp = QLabel('Exposure Time')
        self.slider_exp = QSlider(PyQt5.QtCore.Qt.Horizontal, self)
        self.label_exp_value = QLineEdit()
        self.label_gain_auto = QLabel('Gain Auto')
        self.check_gain_auto = QCheckBox()
        self.check_gain_auto.setTristate(False)
        self.label_gain = QLabel('Gain')
        self.slider_gain = QSlider(PyQt5.QtCore.Qt.Horizontal,self)
        self.label_gain_value = QLineEdit()
        self.label_dn = QLabel('dn')
        self.edit_dn = QLineEdit()
        self.label_wavelength = QLabel('Wavelength (um)')
        self.edit_wavelength = QLineEdit()
        #self.label_filter_size=QLabel('Filter Size')
        #self.edit_filter_size = QLineEdit()
        self.label_mag= QLabel('Mag')
        self.edit_mag=QLineEdit('44.4')
        self.label_pixel_size= QLabel('Pixel Size')
        self.edit_pixel_size=QLineEdit('4.8')
        ##
        self.button_analyze= QPushButton('Analyze')
        #self.button_bg_image = QPushButton('Background')
        self.button_start_convertion = QPushButton('Start_Convertion')
        self.button_stop_convertion = QPushButton('Stop_Convertion')
        self.combo_phase_height = QComboBox()
        self.combo_phase_height.addItems(["Phase", "Height"])
        self.label_colormap_auto= QLabel('Colormap Auto')
        self.check_colormap_auto= QCheckBox()
        self.check_colormap_auto.setCheckState(True)
        self.check_colormap_auto.setTristate(False)
        self.label_colormap_value = QLabel('Colormap Range')
        self.edit_colormap_min= QLineEdit()
        self.label_middle_colormap = QLabel('-')
        self.edit_colormap_max= QLineEdit()


        #self.Phaseimg = PhaseImg()
        self.image = np.array([])
        self.mode=0 # 0 is phase mode, 1 is height mode
        #self.keep_convert=1 # 0 disable convert, 1 enable convert
        self.cam = cam
        self.exp_auto = True
        self.gain_auto = True
        self.exp_time = 0
        self.gain = 0
        self.colormap_auto = 1
        #self.mag=44.4
        #self.pixel_size=4.8
        self.colormap_min = None
        self.colormap_max = None
        colormap = cm.get_cmap("jet")  # cm.get_cmap("CMRmap")
        colormap._init()
        self.lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        self.edit_dn.setText(str(global_variables.dn))
        self.edit_wavelength.setText(str(global_variables.wavelength))
        #self.edit_filter_size.setText(str(self.Phaseimg.radi))'''

        #self.slider = QSlider(Qt.Horizontal)
        #self.slider.setRange(0,10)


        ## setting gridlayout
        self.setting_layout= QGridLayout()
        self.setting_layout.addWidget(self.label_frame_rate,0,0,1,1)
        self.setting_layout.addWidget(self.label_frame_rate_value,0,1,1,2)
        self.setting_layout.addWidget(self.label_exp_auto,1,0,1,1)
        self.setting_layout.addWidget(self.check_exp_auto,1,1,1,2)
        self.setting_layout.addWidget(self.label_exp,2,0,1,1)
        self.setting_layout.addWidget(self.slider_exp,2,1,1,1)
        self.setting_layout.addWidget(self.label_exp_value,2,2,1,1)
        self.setting_layout.addWidget(self.label_gain_auto,3,0,1,1)
        self.setting_layout.addWidget(self.check_gain_auto,3,1,1,2)
        self.setting_layout.addWidget(self.label_gain,4,0,1,1)
        self.setting_layout.addWidget(self.slider_gain,4,1,1,1)
        self.setting_layout.addWidget(self.label_gain_value,4,2,1,1)
        self.setting_layout.addWidget(self.label_dn,5,0,1,1)
        self.setting_layout.addWidget(self.edit_dn,5,1,1,2)
        self.setting_layout.addWidget(self.label_wavelength,6,0,1,1)
        self.setting_layout.addWidget(self.edit_wavelength,6,1,1,2)
        #self.setting_layout.addWidget(self.label_filter_size,7,0,1,1)
        #self.setting_layout.addWidget(self.edit_filter_size,7,1,1,2)
        self.setting_layout.addWidget(self.label_mag,7,0,1,1)
        self.setting_layout.addWidget(self.edit_mag,7,1,1,2)
        self.setting_layout.addWidget(self.label_pixel_size,8,0,1,1)
        self.setting_layout.addWidget(self.edit_pixel_size,8,1,1,2)        
        ##
        self.layout = QVBoxLayout()
        #self.layout.addWidget(self.button_bg_image)
        self.layout.addWidget(self.button_start_convertion)
        self.layout.addWidget(self.button_analyze)
        self.layout.addWidget(self.combo_phase_height)
        ##
        self.view_layout = QHBoxLayout()
        self.graph_view1= GraphicsView()
        self.image_view1= ViewBox()
        self.image_view1.setAspectLocked()
        self.graph_view1.setCentralItem(self.image_view1)
        #self.image_view1 = self.view_layout.addViewBox(row=0, col=0, lockAspect=True)
        self.raw_imageitem = ImageItem()
        self.image_view1.addItem(self.raw_imageitem)
        self.view_layout.addWidget(self.graph_view1)
        self.raw_image_histLUT=HistogramLUTWidget()
        self.raw_image_histLUT.setLevels(0, 255)
        self.view_layout.addWidget(self.raw_image_histLUT)
        self.raw_image_histLUT.setImageItem(self.raw_imageitem)
        #self.image_view1.autoRange()
        self.graph_view2= GraphicsView()
        self.image_view2= ViewBox()
        self.image_view2.setAspectLocked()
        self.graph_view2.setCentralItem(self.image_view2)
        self.converted_imageitem = ImageItem()
        self.image_view2.addItem(self.converted_imageitem)
        self.view_layout.addWidget(self.graph_view2)
        self.converted_image_histLUT=HistogramLUTWidget()
        self.view_layout.addWidget(self.converted_image_histLUT)
        self.converted_image_histLUT.setImageItem(self.converted_imageitem)
        #self.image_view2.autoRange()
        ##
        self.colorlayout=QHBoxLayout()
        self.colorlayout.addWidget(self.label_colormap_auto)
        self.colorlayout.addWidget(self.check_colormap_auto)
        self.colorlayout.addWidget(self.label_colormap_value)
        self.colorlayout.addWidget(self.edit_colormap_min)
        self.colorlayout.addWidget(self.label_middle_colormap)
        self.colorlayout.addWidget(self.edit_colormap_max)
        ##
        self.layout.addLayout(self.view_layout)
        self.layout.addLayout(self.colorlayout)
        self.layout.addWidget(self.button_stop_convertion)
        self.main_layout= QGridLayout()
        self.main_layout.addLayout(self.setting_layout, 0, 0, 8, 2)
        self.main_layout.addLayout(self.layout, 0, 3, 16, 16)
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.tabWidget)
        self.tabWidget.addTab(self.central_widget, 'Viewfinder')


        #self.button_bg_image.clicked.connect(self.obtain_bg_image)
        self.button_start_convertion.clicked.connect(self.convertion)
        self.button_start_convertion.clicked.connect(self.enable_convert)
        self.button_stop_convertion.clicked.connect(self.disable_convert)
        self.button_analyze.clicked.connect(self.analyze_image)
        self.check_exp_auto.clicked.connect(self.change_exp_auto)
        self.slider_exp.valueChanged.connect(self.set_exp)
        self.label_exp_value.returnPressed.connect(self.set_exp_lineedit)
        self.check_gain_auto.clicked.connect(self.change_gain_auto)
        self.slider_gain.valueChanged.connect(self.set_gain)
        self.label_gain_value.returnPressed.connect(self.set_gain_lineedit)
        self.edit_dn.returnPressed.connect(self.change_dn)
        self.edit_wavelength.returnPressed.connect(self.change_wavelength)
        #self.edit_filter_size.returnPressed.connect(self.change_filter_size)
        self.combo_phase_height.currentIndexChanged.connect(self.chose_phase_height)
        self.tabWidget.tabCloseRequested.connect(self.close_tab)
        self.check_colormap_auto.clicked.connect(self.change_colormap_auto)
        self.edit_colormap_min.returnPressed.connect(self.change_colormap_min)
        self.edit_colormap_max.returnPressed.connect(self.change_colormap_max)
        self.edit_mag.returnPressed.connect(self.change_mag)
        self.edit_pixel_size.returnPressed.connect(self.change_pixel_size)
    



        #self.update_timer = QTimer()
        #self.update_timer.timeout.connect(self.update_movie)

        self.start_working()
        #self.timer = QTimer()
        #self.timer.setInterval(7)
        self.threadpool= QThreadPool()
        self.cam.start()
        self.cam.set_buffer_count(1)
        #print(self.cam.get_buffer_count())
        self.run_cam()



    def start_working(self):


        self.label_frame_rate_value.setText(str(self.cam.get_frame_rate()))
        self.exp_auto = self.cam.get_auto_exposure()
        if self.exp_auto:
            self.check_exp_auto.setCheckState(True)
        else:
            self.check_exp_auto.setCheckState(False)
        self.exp_min=int(self.cam.get_exp_min())
        self.exp_max=int(self.cam.get_exp_max())
        self.exp_time = int(self.cam.get_exp())
        self.slider_exp.setValue(self.exp_time)
        self.slider_exp.setRange(self.exp_min,self.exp_max)
        self.gain_auto = self.cam.get_auto_gain()
        if self.gain_auto:
            self.check_gain_auto.setCheckState(True)
        else:
            self.check_gain_auto.setCheckState(False)
        self.gain_min=int(self.cam.get_gain_min())
        self.gain_max=int(self.cam.get_gain_max())
        self.slider_gain.setRange(self.gain_min,self.gain_max)
        self.gain= int(self.cam.get_gain())
        self.slider_gain.setValue(self.gain)



    def run_cam(self):
        aquisition_worker= Acquisition_thread(self.cam)
        aquisition_worker.signals.emit_img.connect(self.update_image)
        aquisition_worker.signals.emit_cam_status.connect(self.update_cam_status)
        self.threadpool.start(aquisition_worker)

    def update_image(self, img):
        #print(img.type)
        self.raw_imageitem.setImage(img, autoLevels=False, levels= (0,255))
        #self.image = img
        #self.run_cam()


    '''def obtain_bg_image (self):
        self.Phaseimg.set_cal_image(self.image)'''


    def update_phase_or_height(self, converted_img):
        #self.converted_imageitem.setImage(converted_img.T, lut = self.lut)

        if (self.colormap_auto==0) and (self.colormap_min!=None) and (self.colormap_max!=None):
            self.converted_imageitem.setImage(converted_img, autoLevels=False, levels=(self.colormap_min,self.colormap_max))
        else:
            self.converted_imageitem.setImage(converted_img)





    def enable_convert(self):
        self.convertion()

    def disable_convert(self):
        self.phase_worker.stop()

    def convertion(self):
        #self.Phaseimg.set_raw_image(self.image)
        self.phase_worker=Image_retrieval(self.mode) 
        self.phase_worker.signals.emit_converted_img.connect(self.update_phase_or_height)
        #if self.keep_convert==1:
        #    phase_worker.signals.finished.connect(self.convertion)
        self.threadpool.start(self.phase_worker)

    def analyze_image(self):
        index=self.tabWidget.count()
        print(index)
        analyzer=Analyzer_Window(self.Phaseimg)
        self.tabWidget.addTab(analyzer, "Capture #{}".format(index))
        self.tabWidget.setCurrentIndex(index)

    '''def change_aquiz_auto(self, s_aquiz):'''


    def chose_phase_height(self, i):
        if i==0:
            self.phase_worker.change_mode(0)
            self.mode=0
        if i==1:
            self.phase_worker.change_mode(1)
            self.mode=1

    def change_dn(self):
        global_variables.dn=float(self.edit_dn.text())

    def change_wavelength(self):
        global_variables.wavelength=float(self.edit_wavelength.text())

    #def change_filter_size(self):
    #    self.Phaseimg.radi=float(self.edit_filter_size.text())

    def update_cam_status(self, cam_status):
        # print(cam_status)
        if self.exp_auto:
            self.slider_exp.setValue(cam_status[1])
        if self.gain_auto:
            self.slider_gain.setValue(cam_status[2])
        self.label_frame_rate_value.setText(str(cam_status[0]))

    def change_exp_auto(self, s_ea):

        if s_ea == True:
            self.cam.set_auto_exposure(True)
            self.exp_auto=True
        elif s_ea == False:
            self.cam.set_auto_exposure(False)
            self.exp_auto=False

    def change_gain_auto(self, s_g):
        if s_g == True:
            self.cam.set_auto_gain(True)
            self.gain_auto=True
        elif s_g == False:
            self.cam.set_auto_gain(False)
            self.gain_auto=False

    def set_gain(self, value):
        #print(value)
        self.label_gain_value.setText("{} dB".format(value))
        if self.gain_auto==False:
            #self.gain=value
            self.cam.set_gain(value)
            #print('changed')

    def set_gain_lineedit(self):
        value=int(self.label_gain_value.text())
        self.slider_gain.setValue(value)
        if self.gain_auto==False:
            #self.gain=value
            self.cam.set_gain(value)   

   

    def set_exp(self, value):
        self.label_exp_value.setText("{} ms".format(value))
        #print(value)
        if self.exp_auto==False:
            #self.exp_time=value
            self.cam.set_exp(value)

    def set_exp_lineedit(self):
        value=int(self.label_exp_value.text())
        self.slider_exp.setValue(value)
        if self.exp_auto==False:
            #self.exp_time=value
            self.cam.set_exp(value)

    def close_tab(self,index):
        if index!=0:
            self.tabWidget.removeTab(index)

    def change_colormap_auto(self, s_ca):
        if s_ca == True:
            self.colormap_auto=1
        elif s_ca == False:
            self.colormap_auto=0
    
    def change_colormap_min(self):
        self.colormap_min=float(self.edit_colormap_min.text())

    def change_colormap_max(self):
        self.colormap_max=float(self.edit_colormap_max.text())
        
    def change_mag(self):
        if self.edit_mag.text()=='':
            self.edit_mag.setText('44.4')
            global_variables.mag=44.4
        else:
            global_variables.mag=float(self.edit_mag.text())
        #self.Phaseimg.zoom_index=self.pixel_size/self.mag

    def change_pixel_size(self):
        if self.edit_pixel_size.text()=='':
            self.edit_pixel_size.setText('4.8')
            global_variables.pixel_size=4.8
        else:
            global_variables.pixel_size=float(self.edit_pixel_size.text())
        #self.Phaseimg.zoom_index=self.pixel_size/self.mag'''




if __name__ == '__main__':
    print(global_variables.dn)
    cam = FLIRCamDev()
    app = QApplication([])
    window = StartWindow(cam)
    window.show()
    app.exit(app.exec_())
    cam.stop()
    cam.close()