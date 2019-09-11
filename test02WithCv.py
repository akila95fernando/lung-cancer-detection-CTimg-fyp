import pydicom as dicom #read dicom file
import os #do directory operations
import pandas as pd
import  cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = "D:\\FYP\\lung cancer cT\\rider images\\RIDER Lung CT\\RIDER-1129164940\\11-06-2014-1-96508\\4-24533\\000156.dcm"
img = dicom.dcmread(image_path)
print(img.PatientID) #read header info on dicom image

#arr = img.pixel_array
#np.set_printoptions(threshold=np.inf)
#print(arr)