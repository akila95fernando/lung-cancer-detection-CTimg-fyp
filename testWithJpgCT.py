#import inline
import matplotlib
#%matplotlib inline
import numpy as np
import cv2
#import skimage
#import os
#import glob
#import dicom as pdicom
from matplotlib import pyplot as plt
#from skimage.morphology import extrema
#from skimage.morphology import watershed as skwater


def ShowImage(title,img,ctype):
  plt.figure(figsize=(10, 10))
  if ctype=='bgr':
    b,g,r = cv2.split(img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    plt.imshow(rgb_img)
  elif ctype=='hsv':
    rgb = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    plt.imshow(rgb)
  elif ctype=='gray':
    plt.imshow(img,cmap='gray')
  elif ctype=='rgb':
    plt.imshow(img)
  else:
    raise Exception("Unknown colour type")
  plt.axis('off')
  plt.title(title)
  plt.show()

#Read in image
imgPath = "D:\\FYP\\testingImgs\\img1.jpg"
img= cv2.imread(imgPath)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#ShowImage('Brain with Skull',gray,'gray')
cv2.namedWindow('Gray', cv2.WINDOW_NORMAL)
cv2.imshow('Gray',gray)

#Make a histogram of the intensities in the grayscale image
plt.hist(gray.ravel(),256)
plt.show()

#Threshold the image to binary using Otsu's method
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
#ShowImage('Threshold Otsu',thresh,'gray')

cv2.imshow('otsu-threshold', thresh )


colormask = np.zeros(img.shape, dtype=np.uint8)
colormask[thresh!=0] = np.array((0,0,255))
blended = cv2.addWeighted(img,0.7,colormask,0.1,0)
#ShowImage('Blended', blended, 'bgr')
cv2.namedWindow('blended', cv2.WINDOW_NORMAL)
cv2.imshow('blended', blended )

ret, markers = cv2.connectedComponents(thresh)

#Get the area taken by each component. Ignore label 0 since this is the background.
marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0]
#Get label of largest component by area
largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above
#Get pixels which correspond to the lung
brain_mask = markers==largest_component

brain_out = img.copy()
#In a copy of the original image, clear those pixels that don't correspond to the brain
brain_out[brain_mask==False] = (0,0,0)
#ShowImage('Connected Components',brain_out,'rgb')
cv2.namedWindow('brainout', cv2.WINDOW_NORMAL)
cv2.imshow('brainout', brain_out )

brain_mask = np.uint8(brain_mask)
kernel = np.ones((8,8),np.uint8)
closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
#ShowImage('Closing', closing, 'gray')
cv2.namedWindow('closing', cv2.WINDOW_NORMAL)
cv2.imshow('closing', closing)







img_median = img.copy()

img_median[closing==False] = (0,0,0)

cv2.namedWindow('final', cv2.WINDOW_NORMAL)
cv2.imshow('final', brain_out )

#median filter
median = cv2.medianBlur(img_median, 5)


cv2.imshow('Median', median)

cv2.waitKey(0)