import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from matplotlib.pyplot import imshow
import sys


for arg in sys.argv:
    print(arg)

file = open("config.txt", 'r')
content = file.read()
paths = content.split("\n") #split it into lines
for path in paths:
    print(path)


path = "/home/pyimagesearch/Desktop/stamp/Test_documents/"
destination = "/home/pyimagesearch/Desktop/stamp/temp/"+ str(image_name.split('.')[0]) +"/"
destination_prob_stamps = "/home/pyimagesearch/Desktop/stamp/temp/"+ str(image_name.split('.')[0]) +"/prob_stamps/"

image_name = arg
img_path = path +  image_name  

if not os.path.exists(destination):
    os.makedirs(destination)

if not os.path.exists(destination_prob_stamps):
    os.makedirs(destination_prob_stamps)    
   
# read image  
img = cv2.imread(img_path, 0)
#plt.imshow(img)

ret, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
cv2.imwrite( destination +"thresh.jpg", thresh )

# connected component labeling and filtering based on size

size_threshold = (img.shape[0]*img.shape[1])
#find all your connected components (white blobs in your image)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(thresh), connectivity=8)
sizes = stats[1:, -1]; nb_components = nb_components - 1

# minimum & maximum size of particles we want to keep (number of pixels)

min_size = 100 
max_size = size_threshold*0.9

img2 = np.zeros((output.shape))
#for every component in the image, you keep it only if it's above min_size & less than max_size
for i in range(0, nb_components):
    if sizes[i] >= min_size and sizes[i] <= max_size:
        img2[output == i + 1] = 255
 
cv2.imwrite( destination+ "connected_components.jpg", (img2) )       
# blurring the image to combine nearby blobs

filter_size = 11 # 21
bb_image = cv2.GaussianBlur(img2,(filter_size,filter_size),0)
# find contours and get the external one
image, contours, hier = cv2.findContours(bb_image.astype('uint8'), cv2.RETR_TREE,
cv2.CHAIN_APPROX_SIMPLE)



# drawing bounding boxes around probable blobs of stamp

# h & w ... based on actual image ratio

stamp_list = []
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    #print(x , " ", y , " ", w , " ",h)
    if (h>img.shape[0]*.1 and h<img.shape[0]*0.8) and (w>img.shape[1]*0.1 and w<img.shape[1]*0.8):# and (w*h)>300:  # 
        #print(x , " ", y , " ", w , " ",h)
        # draw a rectangle to visualize 
        cv2.rectangle(bb_image, (x, y), (x+w, y+h),(255, 255, 0), 2)
        stamp_list.append([y,y+h,x,x+w])

cv2.imwrite( destination+ "blur.jpg", (bb_image) )
# Color filter : filter probable stamps based on color

img = cv2.imread(img_path)

# define the list of boundaries
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
lower_color_spectrum = np.array([0, 100, 100])
upper_color_spectrum = np.array([360, 255, 255])
    
mask = cv2.inRange(hsv, lower_color_spectrum, upper_color_spectrum)
res = cv2.bitwise_and(img,img, mask= mask)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_c = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
size_threshold = (img.shape[0]*img.shape[1])

#find all your connected components (white blobs in your image)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats((mask), connectivity=8)
sizes = stats[1:, -1]; nb_components = nb_components - 1

# minimum & maximum size of particles we want to keep (number of pixels)

min_size = 1 
max_size = size_threshold*0.8

img2 = np.zeros((output.shape))
#for every component in the image, you keep it only if it's above min_size & less than max_size
for i in range(0, nb_components):
    if sizes[i] >= min_size and sizes[i] <= max_size:
        img2[output == i + 1] = 255

filter_size = 101
bb_image = cv2.GaussianBlur(img2,(filter_size,filter_size),0)

# find contours and get the external one
image, contours, hier = cv2.findContours(bb_image.astype('uint8'), cv2.RETR_TREE,
cv2.CHAIN_APPROX_SIMPLE)

# h & w ... based on actual image ratio

#stamp_list = []
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)
    #print(x , " ", y , " ", w , " ",h)
    if (h>img.shape[0]*.05 and h<img.shape[0]*0.9) and (w>img.shape[1]*0.05 and w<img.shape[1]*0.9):# and (w*h)>300:  # 
        #print(x , " ", y , " ", w , " ",h)
        # draw a rectangle to visualize 
        cv2.rectangle(bb_image, (x, y), (x+w, y+h),(255, 255, 0), 2)
        stamp_list.append([y,y+h,x,x+w])
cv2.imwrite( destination+ "color_mask.jpg", (res) )
cv2.imwrite( destination+ "color_bb.jpg", (bb_image) )

for i in range (0,len(stamp_list)):
    extracted_img1 = img_c[stamp_list[i][0]:stamp_list[i][1] ,stamp_list[i][2]:stamp_list[i][3]]
    cv2.imwrite(destination_prob_stamps+str(i)+".png",extracted_img1)












