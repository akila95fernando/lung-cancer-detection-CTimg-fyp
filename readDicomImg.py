import pydicom as dicom #read dicom file
import os #do directory operations
import pandas as pd
import  cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg

image_path = "D:\\FYP\\lung cancer cT\\rider images\\RIDER Lung CT\\RIDER-1129164940\\11-06-2014-1-96508\\4-24533\\000156.dcm"
img = dicom.dcmread(image_path)
print(img.PatientID) #read header info on dicom image
print(img.pixel_array) #read pixel data
print(img.pixel_array.shape)

plt.figure(num='gray scale image') #to show as 2 different figures
img2 = plt.imshow(img.pixel_array, cmap=plt.cm.gray) #color mapping to graylevel


plt.figure(num='bone conversion')
plt.imshow(img.pixel_array, cmap=plt.cm.bone)


plt.show()

# dcm_sample = img.pixel_array
# cv2.imshow("sample_dcm_img",dcm_sample)
# cv2.waitKey(0)
# cv2.destroyAllWindows()









