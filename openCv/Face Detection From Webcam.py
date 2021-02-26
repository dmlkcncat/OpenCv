#!/usr/bin/env python
# coding: utf-8

# # Face Detection From Webcam

# In[8]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[9]:


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")


# In[10]:


def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(frame , (x,y), (x+w, y+h), (255,0,0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
    return frame


# In[11]:


capture = cv2.VideoCapture(0)
while True:
    _, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray,frame)
    cv2.imshow('video', canvas)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break 
capture.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




