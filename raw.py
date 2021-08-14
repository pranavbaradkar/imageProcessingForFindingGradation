import cv2 as cv
import sys
from matplotlib import pyplot as plt
import numpy as np 
import math 
import csv 

speciman_area = 7850
print(speciman_area) 
img1 = cv.imread(cv.samples.findFile("D:/7th sem/BTP/Pranav Images - DBM Images/dbm.png"))
if img1 is None:
    sys.exit("Could not read the image.")
gray1 = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
#median = cv.medianBlur(img,3)
img = cv.fastNlMeansDenoisingColored(img1,None,10,10,7,30)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
median = cv.medianBlur(gray,5)
hist1 = cv.calcHist([gray],[0],None,[256],[0,256])
#plt.hist(gray.ravel(),256,[0,256]); plt.show()
#equ1 = cv.equalizeHist(gray)
clahe = cv.createCLAHE(clipLimit=12.0, tileGridSize=(8,8))
equ1 = clahe.apply(gray)
equ1 = cv.bilateralFilter(equ1,2,200,200)
hist = cv.calcHist([equ1],[0],None,[256],[0,256])
print("Histogram for automatic equalise histogram")
#plt.hist(equ1.ravel(),256,[0,256]); plt.show()
ret,th = cv.threshold(median,100,255,cv.THRESH_BINARY)
ret1,white = cv.threshold(median,0,255,cv.THRESH_BINARY)

contours, hierarchy = cv.findContours(th,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

area = [0] 
count = 1
for cot in contours:
    #M  = cv.moments(cot)
    #print (M['m00'])
   # cx = int (M['m10']/M['m00'])
    #cy = int (M['m01']/M['m00'])
    #center = (cx,cy)
    epsilon = 0.0001 * cv.arcLength(cot,True) 
    approx = cv.approxPolyDP(cot,epsilon,True)
    img1 = cv.drawContours(img1, [approx], -1, (0, 255, 50), 2) 
    area1 = cv.contourArea(cot)
    #print(area1)
    area.append(area1)
    count = count + 1
area.sort() 
print (area)
    
screen_res = 1280, 720
scale_width = screen_res[0] / img.shape[1]
scale_height = screen_res[1] / img.shape[0]
scale = min(scale_width, scale_height)
window_width = int(img.shape[1] * scale)
window_height = int(img.shape[0] * scale)

#cv.namedWindow('dst_rt', cv.WINDOW_NORMAL)
#cv.resizeWindow('dst_rt', window_width, window_height)
#cv.imshow('dst_rt', img1)
#cv.waitKey(0)
ret1,white = cv.threshold(median,0,255,cv.THRESH_BINARY)
n_white_pix_a = np.sum(th==255)   #voids area
print('Number of white pixels aggregate:', n_white_pix_a)
dimensions = gray.shape 
print(dimensions)
print("Height of image",dimensions[0])
print("width of image",dimensions[1])
#5582.37
number_of_total_pixels = np.sum(white==255)
print("total number of pixels",number_of_total_pixels)
mm_square_per_pixel = speciman_area / number_of_total_pixels 
area_of_a = n_white_pix_a * mm_square_per_pixel
print("area of aggregate:",area_of_a)
final_area = [0]
for a in area:
    final_area.append(a*mm_square_per_pixel)
print (final_area)
diameter = [] 
mass = []
cumulative_area = []
cumulative_percentage = []
sum = 0 
for label in final_area:
    sum = sum + label
    cumulative_area.append([label,sum])
    diameter.append(math.sqrt(label * 4 / 3.14))
    mass.append(label) 
for label in cumulative_area : 
    cumulative_percentage.append((label[1]/sum)*100)
batch = [0,0,0,0,0,0,0,0]
print(diameter)
for i in range(0,len(area)+1):
    a = diameter[i] 
    m = mass[i]
    print(a)
    if a < 37.78 : 
        batch[0] = batch[0] + a
    if a < 26.5 : 
        batch[1] = batch[1] + a
    if a < 19 : 
        batch[2] = batch[2] + a
    if a < 13.2 : 
        batch[3] = batch[3] + a
    if a < 4.75 : 
        batch[4] = batch[4] + a
    if a < 2.36 : 
        batch[5] = batch[5] + a 
    if a < 0.3 : 
        batch[6] = batch[6] + a
    if a < 0.075 : 
        batch[7] = batch[7] + a
#batch.sort()    

print(batch) 



#cumulative_percentage = np.array(cumulative_percentage)
#area = np.array(area)
#print (cumulative_percentage,area)
#plt.plot(area,cumulative_percentage) 
#plt.xlabel("area log scale") 
#plt.ylabel('cumulative percentage') 
#plt.grid(True)
#plt.xscale("log")
#plt.title('My first graph!') 
#plt.show() 
#plt.savefig("G:/my_Ml_projects/image processing/Final/rectangular.png")
#plt.show() 
with open("G:/my_Ml_projects/image processing/ResultsD/horizontal_circle/1.csv","w",newline='') as f:
    filename = [["area of a(mm2)"]]
    writer = csv.writer(f)
    writer.writerows(filename)
    writer.writerows(map(lambda x: [x],batch))  