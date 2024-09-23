#! Your Code Goes Here

# Importing the PIL library
from ctypes.wintypes import RGB
import random
import cv2 as cv
import cv2 
import numpy as np
import matplotlib.pyplot as plt

#path = r'C:\Users\DELL\Desktop\Study\Year4 Sem1\Image Processing'
#img_Gray = cv.imread("images/.jpg", cv.IMREAD_GRAYSCALE) 

img = cv.imread("images/img4.jpg") 
inputimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
input_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv2.imwrite('input_img_Gray.png',input_img)

# Add Text to an image
cv2.putText(input_img,"Before",(300,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
#cv.imshow("Gray image before",input_img)

#gamma = 1.2027854
random.seed(12027854)
gamma = random.random()
print("The value of gamma =",gamma,".")
k = 255 / np.power( 255, gamma) 
output_img = k * np.power(inputimg, gamma)
output_img = np.array(output_img, dtype='uint8')
cv2.imwrite('output_img.png',output_img)

# Add Text to an image
cv2.putText(output_img,"After",(300,100),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
#cv.imshow("Gray image after",output_img) 

print("The value of the constant C =",k,".")

hist1 = cv.calcHist([input_img],[0],None,[256],[0,256])
hist2 = cv.calcHist([output_img],[0],None,[256],[0,256])

plt.figure(facecolor= 'g',num="histograms")
plt.subplot(1, 2, 1) # row 1, col 2 index 1
plt.plot(hist1)
plt.title("Histogram Before!")
plt.xlabel("Intensity")
plt.ylabel("Count of Pixels")
plt.xlim([0,256])
plt.locator_params(axis='x',nbins=10)

plt.subplot(1, 2, 2) # index 2
plt.plot(hist2)
plt.title("Histogram after!")
plt.xlabel("Intensity")
plt.ylabel("Count of Pixels")
plt.xlim([0,256])
plt.locator_params(axis='x',nbins=10)
plt.savefig('fig.png')
plt.savefig('12027854.pdf')

hor_img = np.hstack((input_img,output_img))
cv2.imshow('Horizontal_img',hor_img)
hor_img = np.array(hor_img, dtype='uint8')

cv2.imwrite('12027854.png',hor_img)

plt.show()
cv2.waitKey(0)
