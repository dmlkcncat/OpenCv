#!/usr/bin/env python
# coding: utf-8

# # Face Recognition

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[2]:


img = cv2.imread("selfie.jpeg")


# In[3]:


img


# In[4]:


img.shape


# In[5]:


plt.imshow(img)


# In[6]:


gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[7]:


gray_scale


# In[8]:


print(gray_scale.shape)


# In[9]:


print(gray_scale.size)


# In[10]:


print(img.shape)


# In[11]:


print(img.size)


# In[12]:


plt.imshow(gray_scale);


# In[13]:


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray_scale, 1.2, 5)
faces.shape


# In[14]:


faces


# In[15]:


for(x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 4)
cv2.imshow("face detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[21]:


def detectFromImage (image):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    img = cv2.imread(image)
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_scale,1.2, 1)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 4)
    cv2.imshow("baslik", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


# In[22]:


detectFromImage("selfie.jpeg")


# In[19]:


def detectFaces_EyesImage(image):
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    
    img = cv2.imread(image)
    
    gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_scale,1.4, 1)
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 4)
        roi_gray = gray_scale [y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,0,255), 2)
            
    cv2.imshow("Faces and eyes detected.", img)
    cv2.waitKey()
    cv2.destoryAllWindows()


# In[20]:


detectFaces_EyesImage("selfie3.jpg")


# In[ ]:




