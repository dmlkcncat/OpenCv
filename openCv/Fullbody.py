#!/usr/bin/env python
# coding: utf-8

# # Full Body

# In[28]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


# In[57]:



body_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')


# In[58]:


img = cv2.imread('body.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = body_cascade.detectMultiScale(gray, 1.1, 8)

for(x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    
    
cv2.imshow("Faces and eyes detected.", img)
cv2.waitKey()
cv2.destoryAllWindows()


# In[ ]:





# In[ ]:




