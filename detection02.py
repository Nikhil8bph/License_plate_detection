import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pickle
import cv2
import os
import glob

def Th_Bin(image):
    #print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #print(gray.shape)
    #plt.imshow(gray,cmap='gray')
    #plt.show()
    #print(gray.shape)
    ret2,th2 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    #55 for 277
    return th2

def Th_Otsu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray,cmap='gray')
    #plt.show()
    #gray =  cv2.resize(gray,(380,90))
    #plt.imshow(gray,cmap='gray')
    #plt.show()
    img1 = cv2.blur(gray,(2,2),0)
    ret2,th2 = cv2.threshold(img1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th2

def Th_Adap(image):
    #img = cv2.medianBlur(image,5)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #xret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    #th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return th2

def plate(column_list,image):
    print("Loading model")
    filename = './models/svc_model.sav'
    model = pickle.load(open(filename, 'rb'))
    character = []
    for i in os.listdir('temp'):
        im = cv2.imread(os.path.join('temp',i),0)
        img = cv2.resize(im,(20,20))
        ret,imgr = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        binim = imgr.reshape(1,-1)
        n=model.predict(binim)
        character.append(n)
    print(character)
    print(len(character))
    plate_string = ''
    for eachPredict in character:
        plate_string += eachPredict[0]
    print('Predicted license plate')
    print(plate_string)
    column_list_copy = column_list[:]
    column_list.sort()
    rightplate_string = ''
    for each in column_list:
        rightplate_string += plate_string[column_list_copy.index(each)]
    print('License plate')
    print(rightplate_string)
    r = glob.glob('temp/*')
    for i in r:
        os.remove(i)
    os.rmdir('temp')
    path = 'Answers/'+rightplate_string+'.png'
    cv2.imwrite(path,image)


for im in os.listdir('Samples/Nikhil'):
    image = cv2.imread(os.path.join('Samples/Nikhil',im))
    if os.path.exists('temp'):
        print('temp exists')
    else:
        os.mkdir('temp')
    print(os.getcwd())
    imageing = image.copy()
    img = Th_Bin(image)
    license_plate=np.invert(img)
    labelled_plate = measure.label(license_plate)
    #cv2.imshow("license plate",license_plate)
    #cv2.waitKey(100)
    #fig, ax1 = plt.subplots(1)
    #ax1.imshow(image, cmap="gray")
    character_dimensions = (0.30*license_plate.shape[0], 0.60*license_plate.shape[0], 0.03*license_plate.shape[1], 0.15*license_plate.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions
    characters = []
    column_list = []
    plate_no = []
    count = 0
    for regions in regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0
        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            roi = license_plate[y0:y1, x0:x1]
            crop = imageing[y0:y1, x0:x1]
            rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",linewidth=2, fill=False)
            #ax1.add_patch(rect_border)
            resized_char = resize(roi, (20, 20))
            characters.append(resized_char)
            column_list.append(x0)
            pathout = 'temp/'+str(count)+'.jpg'
            cv2.imwrite(pathout,crop)
            count = count+1
    #plt.show()
    plate(column_list,image)