import cv2
import os
import string
from cv2 import resize
images = []

def read():
   count = 0
   if os.path.exists("Test"):
      print("Test folder exists")
   else:
      os.mkdir("Test")
      print("folder created")

   if os.path.exists("output"):
      print('output folder exists')
   else:
      os.mkdir("output")
      print("output folder created")
   #os.mkdir('Test')
   #os.mkdir('output')
   if os.path.exists("Predicted"):
      print('Predicted folder exists')
   else:
      os.mkdir("Predicted")
      print("Predicted folder exist")

   for filename in os.listdir('Samples/Nikhil'):
      img = cv2.imread(os.path.join('Samples/Nikhil',filename))
      img = resize(img,(245*2,135*2))
      cv2.imwrite('Test/frame%d.jpg'%count,img)
      count = count+1

   path = os.getcwd()
   print("Path :- ",path)
   os.chdir("Predicted")
   i=0
   while(i<10):
      mk = str(i)
      os.mkdir(mk)
      i=i+1
   alphabets = string.ascii_uppercase
   for i in alphabets:
      os.mkdir(i)

read()