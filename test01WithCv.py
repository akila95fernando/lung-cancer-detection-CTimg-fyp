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
#print(img.pixel_array) #read pixel data

orginal_img = img.pixel_array
print(orginal_img.shape)

img2 = orginal_img.astype(np.uint8)
img3 = cv2.equalizeHist(img2)
l1Norm = cv2.Canny(img2,100,200,L2gradient=False)
#gray_img = cv2.cvtColor(orginal_img, cv2.COLOR_BGR2GRAY)

cv2.imshow("8 bit image",img2)
cv2.imshow("histoEqlImg",img3)
#plt.figure(num='gray conversion')
#plt.imshow(l1Norm, cmap=plt.cm.gray)


#plt.figure(num='8 bit')
#plt.imshow(img2, cmap=plt.cm.gray)
#plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()