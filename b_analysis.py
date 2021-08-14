import cv2 as cv
import sys
from matplotlib import pyplot as plt
import numpy as np 
import csv
import math
from skimage import color 


img = cv.imread(cv.samples.findFile("G:\my_Ml_projects\image processing\Bituminous Mixture Beam - Size - 30 by 5.jpg"))
if img is None:
    sys.exit("Could not read the image.")
gray1 = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#median = cv.medianBlur(img,5)
median = cv.bilateralFilter(img,2,75,75)
median = cv.bilateralFilter(median,2,75,75)
median = cv.bilateralFilter(median,2,75,75)
gray = cv.cvtColor(median,cv.COLOR_BGR2GRAY)
equ1 = cv.equalizeHist(gray)
hist = cv.calcHist([equ1],[0],None,[256],[0,256])
print("Histogram for automatic equalise histogram")
#plt.hist(equ1.ravel(),256,[0,256]); plt.show()
ret,th = cv.threshold(equ1,140,255,cv.THRESH_BINARY)  #make it 140 for bitumen 7 by 6
res = np.hstack((th,gray1))
cv.imshow("Display window", res) 
cv.waitKey(0)
#plt.imshow(res) 
#plt.title('my picture')
#plt.show()

n_white_pix_a = np.sum(th==255)   #voids area
print('Number of white pixels aggregate:', n_white_pix_a)
dimensions = gray.shape 
print(dimensions)
print("Height of image",dimensions[0])
print("width of image",dimensions[1])
speciman_area = 300 * 50
number_of_total_pixels = dimensions[0] * dimensions[1]
print("total number of pixels",number_of_total_pixels)
mm_square_per_pixel = speciman_area / number_of_total_pixels 
area_of_a = n_white_pix_a * mm_square_per_pixel
print("area of aggregate:",area_of_a)
n_white_pix_bitu = np.sum(th!=255)   #bitumen area
print('Number of white pixels bitu:', n_white_pix_bitu)
area_of_bitu = n_white_pix_bitu * mm_square_per_pixel
print("area of bitumin:",area_of_bitu)


dimensions = gray.shape 
print("Height of image",dimensions[0])
print("width of image",dimensions[1])

num_labels, labels = cv.connectedComponents(th,connectivity=4)
    
# Map component labels to hue val, 0-179 is the hue range in OpenCV
label_hue = np.uint8(179*labels/np.max(labels),axis=1)
blank_ch = 255*np.ones_like(label_hue)
labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
img2 = color.label2rgb(labels,bg_label=0)
# Converting cvt to BGR
labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
sum1 = 0
#for label in labels:
  #  print(label)

# set bg label to black
labeled_img[label_hue==0] = 0
labels = labels.reshape(np.size(gray))
print(np.shape(labels))
label_list = labels.tolist()
label_set = set(label_list)
total_labels = list(label_set)
area_of_each_label = []
for label in total_labels:
    area_of_each_label.append([label,label_list.count(label) * mm_square_per_pixel])
    #print("area of each label image {} = ".format(label),label_list.count(label) * mm_square_per_pixel)
area_of_each_label.sort(key=lambda x: x[-1] )
area_of_each_label.pop()
"""
with open("G:/my_Ml_projects/image processing/area_300_by_50.csv","w") as f:
    filename = [["label","area"]]
    writer = csv.writer(f)
    writer.writerows(filename)
    writer.writerows(area_of_each_label)
"""
area = []
cumulative_area = []
cumulative_percentage = []
sum = 0 
for label in area_of_each_label:
    sum = sum + label[1] 
    cumulative_area.append([label[1],sum])
    area.append(math.sqrt(label[1]))
for label in cumulative_area : 
    cumulative_percentage.append((label[1]/sum)*100)

cumulative_percentage = np.array(cumulative_percentage)
area = np.array(area)

plt.scatter(area,cumulative_percentage) 
plt.xlabel("area log scale") 
plt.ylabel('cumulative percentage') 
plt.grid()
plt.xscale("log")
plt.title('My first graph!') 
#plt.show() 
plt.savefig("G:\my_Ml_projects\image processing\Final\graphBituminous Mixture Beam - Size - 30 by 5.png")





#print(area_of_each_label)
cv.imshow("Display window",img2) 
cv.waitKey(0)
sum1 = 0



"""
# Showing Original Image
plt.imshow(cv.cvtColor(th, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Orginal Image")
plt.show()

#Showing Image after Component Labeling
plt.imshow(cv.cvtColor(labeled_img, cv.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Image after Component Labeling")
plt.show()
"""