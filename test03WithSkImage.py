import pydicom as dicom #read dicom file
import os #do directory operations
import pandas as pd
import  cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from scipy import ndimage as ndi
from glob import glob
from skimage import filters
from skimage import color
from skimage import util
from skimage import morphology,feature

np.set_printoptions(threshold=np.inf) #to show full array in console

image_path = "D:\\FYP\\testingImgs\\000065.dcm"
img = dicom.dcmread(image_path)
print(img.PatientID) #read header info on dicom image
#print(img.pixel_array) #get pixel data

orginal_img = img.pixel_array
converted_img = skimage.util.img_as_ubyte(orginal_img) #convert image 16 bit to u8bit
#print(converted_img)
gray_img = skimage.color.rgb2gray(converted_img) #convert to grayscale
#print(gray_img)

median_img = skimage.filters.median(gray_img) #apply median filter to gray image

thresholds = skimage.filters.threshold_otsu(gray_img) # return threshold values for otsu method
print("thresholds",thresholds)
otsu_img = gray_img > thresholds #apply thresholds and binarize the image
#print(otsu_img)

otsu_median_img = skimage.filters.median(otsu_img)
#print(otsu_median_img)

#watershed segmentation 01 ******************
distance = ndi.distance_transform_edt(otsu_median_img)
local_maxi = skimage.feature.peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                            labels=otsu_median_img)
markers1 = ndi.label(local_maxi)[0]
labels = skimage.morphology.watershed(-distance, markers1, mask=otsu_median_img)
#---**************************************

#watershed segmentation 02 ******************
markers = skimage.filters.rank.gradient(otsu_median_img,skimage.morphology.disk(2))<10
markers = ndi.label(markers)[0]
gradient = skimage.filters.rank.gradient(otsu_median_img, skimage.morphology.disk(2))
#print(gradient)
#---**************************************

all_black_img = otsu_median_img
#for pixel in otsu_median_img:
#    if (pixel==0):
#        pixel = 1

#-----------display images-------------------------------

plt.figure(num='orginal image')
plt.imshow(orginal_img)

plt.figure(num='converted image')
plt.imshow(converted_img, cmap='gray')

#plt.figure(num='gray image')
#plt.imshow(gray_img, cmap='gray')

plt.figure(num='median')
plt.imshow(median_img, cmap='gray')

#plt.figure(num='otsu')
#plt.imshow(otsu_img, cmap='gray')

plt.figure(num='otsuMedian')
plt.imshow(otsu_median_img, cmap='gray')

plt.figure(num='segmented')
plt.imshow(labels, cmap=plt.cm.nipy_spectral)

plt.figure(num='local gradient')
plt.imshow(gradient, cmap='nipy_spectral')

#plt.figure(num='markers')
#plt.imshow(markers, cmap='nipy_spectral')

#plt.figure(num='markers1')
#plt.imshow(markers1, cmap='nipy_spectral')

#plt.figure(num='subImage')
#plt.imshow(otsu_median_img-gradient, cmap='gray')
plt.show()
