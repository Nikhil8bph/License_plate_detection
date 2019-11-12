import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import cv2
#import Ready_set
import matplotlib.pyplot as plt
def segmentadap(image,counter):
    #img = Th_Bin(image)
    #img = Th_Otsu(image)
    img = Th_Adap(image)
    #cv2.imshow("img",img)
    license_plate=np.invert(img)
    labelled_plate = measure.label(license_plate)
    #cv2.imshow("license plate",license_plate)
    #cv2.waitKey(0)
    #fig, ax1 = plt.subplots(1)
    #ax1.imshow(image, cmap="gray")
    character_dimensions = (0.30*license_plate.shape[0], 0.60*license_plate.shape[0], 0.03*license_plate.shape[1], 0.20*license_plate.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions
    characters = []
    column_list = []
    count = 0
    for regions in regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0

        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            roi = license_plate[y0:y1, x0:x1]
            crop = image[y0:y1, x0:x1]
            rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",linewidth=2, fill=False)
            #ax1.add_patch(rect_border)
            resized_char = resize(roi, (20, 20))
            characters.append(resized_char)
            column_list.append(x0)
            pathout = 'output/'+str(counter)+'_'+str(count)+'.jpg'
            cv2.imwrite(pathout,crop)
            count = count+1

    #plt.show()
    if (count <5):
        segmentotsu(image,counter)

def segmentotsu(image,counter):
    img = Th_Otsu(image)
    #img = Th_Adap(image)
    #cv2.imshow("img",img)
    license_plate=np.invert(img)
    labelled_plate = measure.label(license_plate)
    #cv2.imshow("license plate",license_plate)
    #cv2.waitKey(0)
    #fig, ax1 = plt.subplots(1)
    #ax1.imshow(image, cmap="gray")
    character_dimensions = (0.20*license_plate.shape[0], 0.70*license_plate.shape[0], 0.01*license_plate.shape[1], 0.35*license_plate.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions
    characters = []
    column_list = []
    count = 0
    for regions in regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0

        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            roi = license_plate[y0:y1, x0:x1]
            crop = image[y0:y1, x0:x1]
            rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",linewidth=2, fill=False)
            #ax1.add_patch(rect_border)
            resized_char = resize(roi, (20, 20))
            characters.append(resized_char)
            column_list.append(x0)
            pathout = 'output/'+str(counter)+'_'+str(count)+'.jpg'
            cv2.imwrite(pathout,crop)
            count = count+1

    #plt.show()

def Th_Adap(image):
    #img = cv2.medianBlur(image,5)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #xret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    #th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return th2

def Th_Otsu(image):
    #print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(gray.shape)
    #plt.imshow(gray,cmap='gray')
    #plt.show()
    #gray =  cv2.resize(gray,(380,90))

    #plt.imshow(gray,cmap='gray')
    #plt.show()
    print(gray.shape)
    img1 = cv2.blur(gray,(2,2),0)
    ret2,th2 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th2


def runner():
    path = 'Test'
    counter = 0
    import os
    for i in os.listdir(path):
        #crawl = path + '/frame' +i+'.jpg'
        image = cv2.imread(os.path.join('Test',i))
        segmentadap(image,counter)
        counter = counter+1



runner()