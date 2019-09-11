import cv2

imgPath = "D:\\FYP\\lung cancer cT\\rider images\\RIDER Lung CT\\RIDER-1129164940\\11-06-2014-1-96508\\4-24533\\000156.dcm"
#imgPath = "D:\\FYP\\lung cancer cT\\rider images\\RIDER Lung CT\\RIDER-1129164940\\11-06-2014-1-96508\\4-24533\\000090.dcm"
img = cv2.imread(imgPath,0) #0 to frayscale 1 to color
#resizedImg = cv2.resize(img,(600,600)) #(image,(width,height)
resizedImg = cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2))) #reduce image size ny 4 times

l1Norm = cv2.Canny(resizedImg,10,200,L2gradient=False)


finalSubImg = cv2.subtract(resizedImg,l1Norm)
finalAddImg = cv2.add(resizedImg,l1Norm)
#print(img.shape)
cv2.imshow("imageWindow",resizedImg)
cv2.imshow("edgeImageWindow",l1Norm)
cv2.imshow("subImageWindow",finalSubImg)
cv2.imshow("addImageWindow",finalAddImg)
print(resizedImg.shape)
print(l1Norm.shape)
print(finalSubImg.shape)

print(resizedImg)
print(l1Norm)
print(finalSubImg)

cv2.waitKey(0)
cv2.destroyAllWindows()