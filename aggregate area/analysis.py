#images with cropped surface
import cv2 as cv
import sys
from matplotlib import pyplot as plt
import numpy as np 
import csv
x = [] 
xi = []
temp = []
for i in range(1,14):
    image_path = "G:\my_Ml_projects\image processing\p{}.PNG".format(i)
    img = cv.imread(cv.samples.findFile(image_path))
    if img is None:
        sys.exit("Could not read the image.")
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #cv.imshow("Display window", gray)
    #cv.waitKey(0)
    dimensions = gray.shape 
    print(dimensions)
    print("Height of image",dimensions[0])
    print("width of image",dimensions[1])
    equ1 = cv.equalizeHist(gray)
    clahe = cv.createCLAHE(clipLimit=12.0, tileGridSize=(8,8))
    equ = clahe.apply(gray)
    #blur = cv.bilateralFilter(equ,5,51,51)
    #blur = cv.GaussianBlur(img,(5,5),0)
    #ret,th3 = cv.threshold(equ,255,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ret,th = cv.threshold(equ,210,255,cv.THRESH_BINARY)   #for cement paste 
    ret,th1 = cv.threshold(equ1,50,255,cv.THRESH_BINARY_INV) # for air voids 
    ret,th2 = cv.threshold(equ1,0,255,cv.THRESH_BINARY) #whole image
    #adding th and th1 will give image for aggregate 
    dst1 = cv.addWeighted(th,1,th1,1,0)
    ret,dst = cv.threshold(dst1,100,255,cv.THRESH_BINARY_INV) 
    res = np.hstack((th,th1,dst,gray))
    #cv.imshow("Display window", res) 
    #cv.waitKey(0)
    plt.imshow(res) 
    plt.title('my picture')
    #plt.show()
    hist = cv.calcHist([equ1],[0],None,[256],[0,256])
    print("Histogram for automatic equalise histogram")
    #plt.hist(equ1.ravel(),256,[0,256]); plt.show()
    hist = cv.calcHist([equ],[0],None,[256],[0,256])
    print("Histogram for clahe equalisation")
    #plt.hist(equ.ravel(),256,[0,256]); plt.show()
    hist = cv.calcHist([th],[0],None,[256],[0,256])
    print("Histogram for cement paste binary")
    #plt.hist(th.ravel(),256,[0,256]); plt.show()
    hist = cv.calcHist([th1],[0],None,[256],[0,256])
    print("Histogram for air voids binary")
    #plt.hist(th1.ravel(),256,[0,256]); plt.show()
    hist = cv.calcHist([dst],[0],None,[256],[0,256])
    print("Histogram for aggregate binary")
    #plt.hist(dst.ravel(),256,[0,256]); plt.show()
    #hist = cv.calcHist([th2],[0],None,[256],[0,256])
    #plt.hist(th2.ravel(),256,[0,256]); plt.show()
    
    edges = cv.Canny(th1,150,255)
    edges1 = cv.Canny(th,150,255)
    edges2 = cv.Canny(dst,150,255)
    final = np.hstack((edges,th1))
    final1 = np.hstack((edges1,th))
    final2 = np.hstack((edges2,dst))
    #cv.imshow("air voids", final) 
    #cv.imshow("cement paste", final1) 
    #cv.imshow("agregate", final2) 
    #cv.waitKey(0)
    n_white_pix_w = np.sum(th2==255)  #whole area
    #print('Number of white pixels whole:', n_white_pix_w)
    xi.append(image_path)
    xi.append(n_white_pix_w)
    n_white_pix_v = np.sum(th1==255)   #voids area
    #print('Number of white pixels voids:', n_white_pix_v)
    xi.append(n_white_pix_v)
    n_white_pix_c = np.sum(th==255)     #cement area
    #print('Number of white pixels cement:', n_white_pix_c)
    xi.append(n_white_pix_c)
    n_white_pix_a = np.sum(dst==255)  #aggregate area
    #print('Number of white pixels aggregate:', n_white_pix_a)
    xi.append(n_white_pix_a)
    speciman_area = 100 * 200 
    number_of_total_pixels = dimensions[0] * dimensions[1]
    mm_square_per_pixel = speciman_area / number_of_total_pixels 
    area_of_v = n_white_pix_v * mm_square_per_pixel
    area_of_c = n_white_pix_c * mm_square_per_pixel
    area_of_a = n_white_pix_a * mm_square_per_pixel
    area_whole = area_of_v + area_of_c + area_of_a
    xi.append(area_of_v)
    xi.append(area_of_c)
    xi.append(area_of_a)
    xi.append(area_whole)
    temp = xi.copy()
    x.append(temp) 
    xi.clear()
with open("G:/my_Ml_projects/image processing/result.csv","w") as f:
    filename = [["image_path","Number of white pixels whole","Number of white pixels voids","Number of white pixels cement","Number of white pixels aggregate","area of v(mm2)","area of c(mm2)","area of a(mm2)","area of whole(mm2)"]]
    writer = csv.writer(f)
    writer.writerows(filename)
    writer.writerows(x)

    f.close()



