import cv2 as cv
import sys
from matplotlib import pyplot as plt
import numpy as np 
import math 


cumulative_percentage = [1,2,2,2,8]
area = [6.66,20,33.33,46.66,100]
cumulative_percentage = np.array(cumulative_percentage)
area = np.array(area)
print (cumulative_percentage,area)
plt.plot(area,cumulative_percentage) 
plt.xlabel("area log scale") 
plt.ylabel('cumulative percentage') 
plt.grid(True)
plt.xscale("log")
plt.title('My first graph!') 
#plt.show() 
#plt.savefig("G:\my_Ml_projects\image processing\Final\try.png")
plt.show() 