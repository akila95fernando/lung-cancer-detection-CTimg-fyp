import pydicom as dicom #read dicom file
import os #do directory operations
import pandas as pd
import  cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
import re
from scipy import ndimage as ndi
from glob import glob
from skimage import filters
from skimage import color
from skimage import util
from skimage import morphology,feature
from skimage import io

#np.set_printoptions(threshold=np.inf) #to show full array in console

inputImgFolder_path = "D:\\FYP\\test01\\sampleImgs"
#print(glob(inputImgFolder_path+"/*.dcm"))
inputFileList = glob(inputImgFolder_path+"/*.dcm")
#print(inputFileList)

outputImgFolder_path = "D:\\FYP\\test01\\outputImgs"

def getProcessedImg(imgPath):
    #image_path = imgPath
    img = dicom.dcmread(imgPath)
    orginal_img = img.pixel_array
    converted_img = skimage.util.img_as_ubyte(orginal_img)    #convert image 16 bit to u8bit
    gray_img = skimage.color.rgb2gray(converted_img)          #convert to grayscale
    median_img = skimage.filters.median(gray_img)             #apply median filter to gray image
    thresholds = skimage.filters.threshold_otsu(gray_img)     #return threshold values for otsu method
    otsu_img = gray_img > thresholds                          #apply thresholds and binarize the image
    otsu_median_img = skimage.filters.median(otsu_img)

    #watershed segmentation 02 ******************
    markers = skimage.filters.rank.gradient(otsu_median_img,skimage.morphology.disk(2))<10
    markers = ndi.label(markers)[0]
    gradient = skimage.filters.rank.gradient(otsu_median_img, skimage.morphology.disk(2))
    #---*****************************************

    subImage = otsu_median_img-gradient
    return subImage

out_prefix = "output"
out_postfix = ".png"

for imgPath in inputFileList: #loop over every input image
    processsed_img = getProcessedImg(imgPath)                                  #method call for transformation
    out_name_arr = re.findall('\d+', imgPath)                                  #regex to extract input image name index
    out_name = out_name_arr[1]                                                 #second array element is the index
    outputImg_name = out_prefix + str(out_name) + out_postfix                  #output image name
    skimage.io.imsave(outputImgFolder_path+"\\"+outputImg_name,processsed_img) #save output image to folder

    