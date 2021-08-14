#analysis for Bituminous Mixture Beam - Size - 30 by 5.jpg

import cv2 as cv
import sys
from matplotlib import pyplot as plt
import numpy as np 


img1 = cv.imread(cv.samples.findFile("G:\my_Ml_projects\image processing\circle-cropped.png"))
if img1 is None:
    sys.exit("Could not read the image.")
gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
#median = cv.medianBlur(img,3)
img = cv.fastNlMeansDenoisingColored(img1,None,10,10,7,30)
median = cv.bilateralFilter(img,2,100,100)
median = cv.bilateralFilter(median,2,100,100)
median = cv.bilateralFilter(median,2,100,100)
median = cv.bilateralFilter(median,2,100,100)
median = cv.bilateralFilter(median,2,100,100)
gray = cv.cvtColor(median,cv.COLOR_BGR2GRAY)
hist1 = cv.calcHist([gray],[0],None,[256],[0,256])
plt.hist(gray.ravel(),256,[0,256]); plt.show()
equ1 = cv.equalizeHist(gray)
hist = cv.calcHist([equ1],[0],None,[256],[0,256])
print("Histogram for automatic equalise histogram")
plt.hist(equ1.ravel(),256,[0,256]); plt.show()
ret,th = cv.threshold(equ1,140,255,cv.THRESH_BINARY)
edges = cv.Canny(th,170,255)

kernel = np.ones((5,5), np.uint8) 
#img_dilation = cv.dilate(th, kernel, iterations=2) 
#img_erosion = cv.erode(th, kernel, iterations=2) 
#img_dilation = cv.dilate(img_erosion, kernel, iterations=2) 
opening = cv.morphologyEx(th, cv.MORPH_OPEN, kernel,iterations=1) #opening
sure_bg = cv.dilate(opening,kernel,iterations = 1) 
dist_tr = cv.distanceTransform(opening,cv.DIST_L2,3)
ret1,th1 = cv.threshold(dist_tr,0.1*dist_tr.max(),255,0)
th1 = np.uint8(th1) 
unknown = cv.subtract(sure_bg,th1)
ret2,markers = cv.connectedComponents(th1)
markers = markers + 10 
markers[unknown == 255] = 0 

#img_dilation = cv.morphologyEx(th, cv.MORPH_CLOSE, kernel,iterations=1 )
#opening = cv.morphologyEx(th,cv.MORPH_OPEN,kernel, iterations = 2)
#dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
#ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#sure_fg = np.uint8(sure_fg)
#screen_res = 1280, 720
#scale_width = screen_res[0] / img.shape[1]
#scale_height = screen_res[1] / img.shape[0]
#scale = min(scale_width, scale_height)
#window_width = int(img.shape[1] * scale)
#window_height = int(img.shape[0] * scale)

#cv.namedWindow('dst_rt', cv.WINDOW_NORMAL)
#cv.resizeWindow('dst_rt', window_width, window_height)
#res = np.hstack((th,unknown,markers))
#cv.imshow('dst_rt', markers)
#cv.waitKey(0)
#res = np.hstack((gray,th,edges))
plt.imshow(markers) 
##plt.title('my picture')
##plt.show()
