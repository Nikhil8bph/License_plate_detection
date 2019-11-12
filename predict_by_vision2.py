import os
import cv2
import io
import numpy as np
import string
#import segment
#import Ready_set
list = []
index = []
def vision(img):
    from google.cloud import vision
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="dog.json"
    client = vision.ImageAnnotatorClient()
    with io.open(img, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')
    l = []
    for text in texts:
        print('\n"{}"'.format(text.description))
        l = text.description

    print(l)
    if len(l)==2:
        list.append(l[0])
        #list.append(l[1])
    else:
        list.append('nan')
        #list.append('nan')


def process(l1,a):
    image1 = cv2.imread(('output/'+l1),0)
    image1 = cv2.resize(image1,(200, 200), None, .25, .25)
    numpy_horizontal_concat = np.concatenate((image1, image1), axis=1)
    path = 'trash1/'+str(a)+'.jpg'
    cv2.imwrite(path,numpy_horizontal_concat)

def call():
    list_output = []
    for i in os.listdir('output'):
        list_output.append(i)
    n = len(list_output)
    count = 0
    for l in range(0,n):
        process(list_output[l],count)
        count = count+1

def visioner():
    counter = 0
    for i in os.listdir('trash1'):
        print("The count value is : ",counter)
        path = 'trash1/'+str(counter)+'.jpg'
        vision(path)
        counter = counter+1

count = 0

for i in os.listdir('output'):
    img = cv2.imread(os.path.join('output',i))
    index.append(img)

call()
visioner()
print("length of LIST : ",len(list))
print("Length of Index : ",len(index))

import pandas as pd
df = pd.DataFrame(list)
df.to_csv('List1.csv')

