# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 21:03:18 2020

@author: Mahmut Can Gönül
"""

from imageai import Detection
import cv2

model_path = r"C:\Users\ObjectDetection/yolo.h5"

yolo = Detection.ObjectDetection()

yolo.setModelTypeAsYOLOv3()
yolo.setModelPath(model_path)
yolo.loadModel()


cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH,1300)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,1500)
a = 0
b = 0
total = 0
objects = ["person","book","water glass","toothbrush"]
while True:
    ret,img = cam.read()
    
    
    img,preds = yolo.detectCustomObjectsFromImage(input_image=img, 
                      custom_objects=None, input_type="array",
                      output_type="array",
                      minimum_percentage_probability=70,
                      display_percentage_probability=True,
                      display_object_name=True)
    
    cv2.imshow("",img)
    for eachobject in preds:
            if eachobject["name"] == "person" and eachobject["percentage_probability"] > 0.90:
              a+=1
              print(a)
            if eachobject["name"] == "book" and eachobject["percentage_probability"] > 0.60:
              b+=1
              
            
    total = a + b
        
    
    
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break
    


print(total)
cam.release()
cv2.DestroyAllWindows()


    
    
    
    
    


















        
        
























