import pandas as pd
import os
import cv2
import warnings
warnings.filterwarnings(action='ignore')
#import predict_by_vision2
#import Ready_set
def match(text,image,counter):
    s = str(text)
    print(s)
    dir = os.listdir('Predicted')

    if s in dir:
        if s!='nan':
            sub_dir = os.listdir('Predicted/'+s)
            count = 0
            for j in sub_dir:
                count = count+1
            for i in dir:
                if (i == text):
                    path = 'Predicted/'+s+'/'+s+'_'+str(count)+'.jpg'
                    img = cv2.resize(image,(20,20))
                    cv2.imwrite(path,img)
                    print("I have written : ",text)
                    counter = counter+1
                elif (text == 'А'):
                    print(' ')
                else:
                    print(' ')
    else:
        print("This dosen't exist")

    '''
    print("n is : ",n)
    print(t)
    print(s)
    print(os.getcwd())
    if(text!=None):
        path = 'Predicted/'+s+'/'+s+'_'++'.jpg'
        cv2.imwrite(path,cv2.resize(image,(20,20)))
    else:
        print("dosen't match")
        path = 'False/'+str(count)+'.jpg'
        cv2.imwrite(path,image)
    '''


filename ='List1.csv'
hnames = ['pred']
dataframe = pd.read_csv(filename,names=hnames)
print(dataframe.shape)

x = []
y = []
image = []
#ti = hnames['pred']
len = len(os.listdir('output'))
for filename in os.listdir('output'):
    img = cv2.imread(os.path.join('output',filename))
    #path = 'Predicted/'+dataframe['pred']+'/'+dataframe['pred']+'_'+str(len(os.listdir('Predicted/'+i))+1)+'.jpg'
    image.append(img)


#ti = hnames['index']
#print("ti : ",ti[2])
count = 0
counter = 0
y= dataframe['pred'].copy()
for i,j in zip(image,dataframe['pred']):
    x.append(image)
    match(j,i,counter)
    print("Count = ",count)
    count = count+1


print("number of elements predicted is : ", counter)