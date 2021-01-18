from PyQt5.QtWidgets import *
from pyqtgraph import *
from PyQt5.QtCore import *
import pyqtgraph.opengl as gl
from PhaseImage import *
import cv2
from matplotlib import cm


class Analyzer_Window(QWidget):
    def __init__(self, PhaseImage):
        super(Analyzer_Window, self).__init__()

        self.PhaseImage=PhaseImage
        self.phase_or_height = 0         # 0 is phase, 1 is height
        self.magz=1
        self.edit_magz = QLineEdit('1')
        self.a=0.2
        self.calculate_dry_mass= False
        #self.lut=lut
        self.mainLayout = QGridLayout()
        self.origin_GraphicsView= GraphicsView()
        self.origin_ViewBox=ViewBox()
        self.origin_GraphicsView.setCentralItem(self.origin_ViewBox)
        self.origin_image = ImageItem()
        self.origin_ViewBox.addItem(self.origin_image)
        self.origin_image.setImage(PhaseImage.phase_image)
        self.volume_roi = RectROI([300,300],[100,100], pen = (0,60))
        self.origin_ViewBox.addItem(self.volume_roi)
        self.cross_roi = LineROI ([400, 400], [1, 1], width=1 ,pen=(1,60))
        self.origin_ViewBox.addItem(self.cross_roi)
        self.drymass_roi = RectROI([100,100], [70,70],pen = (2,9))
        self.origin_ViewBox.addItem(self.drymass_roi)
        self.mainLayout.addWidget(self.origin_GraphicsView, 1, 0, 8, 1)
        self.image_histLUT=HistogramLUTWidget()
        self.mainLayout.addWidget(self.image_histLUT, 1, 1, 8, 1)
        self.image_histLUT.setImageItem(self.origin_image)
        self.combo_phase_height = QComboBox()
        self.combo_phase_height.addItems(["Phase", "Height"])
        self.mainLayout.addWidget(self.combo_phase_height, 0,0,1,2)



        self.vol_ViewBox = gl.GLViewWidget()
        #self.vol_ViewBox.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        self.vol_ViewBox.setMinimumSize(90, 400)
        #self.vol_ViewBox.setCameraPosition(distance=50)
        self.g=gl.GLGridItem()
        #self.g.scale(2,2,1)
        #self.g.setDepthValue(10)
        #self.vol_ViewBox.addItem(self.g)
        self.plot_vol = gl.GLSurfacePlotItem()
        self.vol_ViewBox.addItem(self.plot_vol)
        self.draw_surface()
        self.mainLayout.addWidget(self.vol_ViewBox, 2, 2, 2, 1)
        self.label_volume_show= QLabel('Volume Show')
        self.mainLayout.addWidget(self.label_volume_show, 0,2,1,1)
        self.layout_magz = QHBoxLayout()
        self.label_magz = QLabel('z mag:')
        self.button_volume_threshorld = QPushButton('Apply Threshold')
        #self.edit_magz = QLineEdit('1')
        self.layout_magz.addWidget(self.label_magz)
        self.layout_magz.addWidget(self.edit_magz)
        self.layout_magz.addWidget(self.button_volume_threshorld)
        self.mainLayout.addLayout(self.layout_magz,1,2,1,1)

        self.label_cross_section= QLabel('Cross Section')
        self.mainLayout.addWidget(self.label_cross_section, 4,2,1,1)
        self.cross_section=PlotWidget()
        self.p1 = self.cross_section.plot()
        self.p1.setPen((200, 200, 100))
        self.cross_section.enableAutoScale()
        self.mainLayout.addWidget(self.cross_section, 5, 2, 2, 1)
        self.layout_dry_mass=QHBoxLayout()
        self.label_calculate_Dry_mass=QLabel('Dry Mass')
        self.check_Dry_mass=QCheckBox()
        self.check_Dry_mass.setCheckState(False)
        self.label_a=QLabel('a=')
        self.edit_a=QLineEdit('0.2')
        self.label_a_unit=QLabel('mL/g')
        self.layout_dry_mass.addWidget(self.label_calculate_Dry_mass)
        self.layout_dry_mass.addWidget(self.check_Dry_mass)
        self.layout_dry_mass.addWidget(self.label_a)
        self.layout_dry_mass.addWidget(self.edit_a)
        self.layout_dry_mass.addWidget(self.label_a_unit)
        self.mainLayout.addLayout(self.layout_dry_mass, 7, 2, 1, 1)
        self.Label_Dry_mass= QLabel('NA')
        self.mainLayout.addWidget(self.Label_Dry_mass, 8, 2, 1, 1)


        self.setLayout(self.mainLayout)

        self.volume_roi.sigRegionChangeFinished.connect(self.draw_surface)
        self.cross_roi.sigRegionChangeFinished.connect(self.draw_cross_section)
        self.cross_section.setLabel('left', 'Phase', units='rad')
        self.cross_section.setLabel('bottom', 'Distance', units='um')
        self.drymass_roi.sigRegionChangeFinished.connect(self.update_dry_mass)
        self.combo_phase_height.currentIndexChanged.connect(self.chose_phase_height)
        self.edit_magz.returnPressed.connect(self.change_mag)
        self.check_Dry_mass.clicked.connect(self.whether_cal_dry_mass)
        self.edit_a.returnPressed.connect(self.change_a)
        #self.image_histLUT.enableMouse()
        #self.image_histLUT.sigMouseReleased.connect(self.draw_surface)
        self.button_volume_threshorld.clicked.connect(self.draw_surface)


        self.draw_surface()
        self.draw_cross_section()
        self.update_dry_mass()


    def draw_surface(self):
        a=self.image_histLUT.getLevels()
        if self.edit_magz.text()=='':
            self.magz=1
        else:
            self.magz= float(self.edit_magz.text())
        if self.phase_or_height == 1:
            target_image=self.PhaseImage.height_image
        else:
            target_image=self.PhaseImage.phase_image
        self.vol_ViewBox.removeItem(self.plot_vol)
        area=self.volume_roi.getArrayRegion(target_image, self.origin_image)
        area[area<a[0]]=a[0]
        area[area>a[1]]=a[1]
        area=area-a[0]
        area_width=area.shape[0]
        area_height=area.shape[1]
        x=np.linspace(-area_width/2, area_width/2, area_width)
        y=np.linspace(-area_width/2, area_width/2, area_height)
        self.plot_vol = gl.GLSurfacePlotItem(x*self.PhaseImage.zoom_index,y*self.PhaseImage.zoom_index,area*self.magz, shader='normalColor')
        #self.plot_vol.translate(-10,-10,0)
        self.vol_ViewBox.addItem(self.plot_vol)


    def draw_cross_section(self):
        if self.phase_or_height == 1:
            target_image=self.PhaseImage.height_image
        else:
            target_image=self.PhaseImage.phase_image
        line=self.cross_roi.getArrayRegion(target_image, self.origin_image)
        num=line.shape[0]
        x=np.linspace(0, num-1, num) * self.PhaseImage.zoom_index
        self.p1.setData(x, line[:,0])



    def update_dry_mass(self):
        if self.calculate_dry_mass:
            d_area=self.volume_roi.getArrayRegion(self.PhaseImage.phase_image, self.origin_image)
            sum_phase=np.sum(d_area)
            dry_mass=sum_phase*self.PhaseImage.lamda/2/np.pi*self.PhaseImage.zoom_index*self.PhaseImage.zoom_index/self.a
            self.Label_Dry_mass.setText("{} pg".format(dry_mass))
        else:
            self.Label_Dry_mass.setText("NA")

    def chose_phase_height(self, i):
        if i==0:
            self.origin_image.setImage(self.PhaseImage.phase_image)
            self.cross_section.setLabel('left', 'Phase', units= 'rad')
            self.phase_or_height = 0
            self.draw_surface()
            self.draw_cross_section()
        if i==1:
            self.origin_image.setImage(self.PhaseImage.height_image)
            self.cross_section.setLabel('left', 'Height', units= 'um')
            self.phase_or_height = 1
            self.draw_surface()
            self.draw_cross_section()
    
    def change_mag(self):
        self.draw_surface()


    def change_a(self):
        if self.edit_magz.text()=='':
            value=0.2
        else:
            value=float(self.edit_a.text())
        self.a=value


    def whether_cal_dry_mass(self, s_wcdm):
        if s_wcdm==True:
            self.calculate_dry_mass=True
            self.update_dry_mass()
        if s_wcdm==False:
            self.calculate_dry_mass=False



if __name__ == '__main__':
    app=QtGui.QApplication([])
    P_image=PhaseImg()
    P_image.set_cal_image(cv2.imread('1009/bg.bmp', 0))
    P_image.set_raw_image(cv2.imread('1009/sample.bmp',0))
    P_image.calculate_phase()
    colormap = cm.get_cmap("jet")  # cm.get_cmap("CMRmap")
    colormap._init()
    lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
    w=Analyzer_Window(P_image)
    w.show()
    QtGui.QApplication.instance().exec_()
