import numpy as np 
from functions import segmentation, phase_retrieval
import os
import cv2
import matplotlib.pyplot as plt

source_dir='RBC100/'
bg_image='bg.tif'
bg_dir=source_dir+bg_image
files=[f for f in os.listdir(source_dir) if os.path.isfile(source_dir+f) and f!=bg_image]
bg_image=cv2.imread(bg_dir)
m=bg_image.shape
merged_image=np.zeros([m[0],m[1],5])
i=0
for f in files[0:5]:
    sample_dir=source_dir+f
    retrived_image=phase_retrieval(sample_dir,bg_dir)
    merged_image[:,:,i]=retrived_image
    i=i+1
#average_image=np.sum(merged_image, axis=2)/len(files)
average_image=np.sum(merged_image, axis=2)/5
plt.imshow(average_image)
plt.colorbar()
plt.show()
mask=segmentation(-average_image, [-0.5,2.5]) # Sometimes the phase is negative, at this time a negative sign is required.
cell_num= np.max(mask)
fluctuation_all=np.std(merged_image, axis=2)
for cell in range(cell_num):
    cell_area=(mask==cell+1)
    if np.sum(cell_area)>0:
        cell_coords=np.argwhere(cell_area)
        fluctuation_cell=fluctuation_all*cell_area
        fluctuation_level=np.sum(fluctuation_cell)/np.sum(cell_area)
        print("Cell {} has membrane fluctuation level of {}".format(str(cell+1),str(fluctuation_level)))
        x_min,y_min=cell_coords.min(axis=0)
        x_max,y_max=cell_coords.max(axis=0)
        plt.imshow(fluctuation_cell[x_min:x_max+1, y_min:y_max+1],cmap=plt.cm.jet)
        plt.colorbar()
        plt.show()