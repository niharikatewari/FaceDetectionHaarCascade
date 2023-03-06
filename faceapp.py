# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:37:53 2023

@author: Lab Pc
"""

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

image = Image.open(r'imagefamily.jpg')
haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
col1, col2  = st.columns([0.8,0.2])

with col1:
    st.markdown("""<style> .font{
        font-size:35px; font-family:'Times New Roman'; color:##FF9633;}
        </style>""", unsafe_allow_html=True)
    st.markdown('Upload your Photo For Face Detection')

with col2:
    st.image(image, width=150)
    
st.sidebar.markdown("Face Detection App")
with st.sidebar.expander("About Face Detection App"):
    st.write("""
             This app detects Faces in the image using HaarCascadeClassifier. Try out!
             """)
uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([0.5,0.5])
    
    with col1:
        st.markdown('Uploaded image')
        st.image(image,width=300)
    with col2:
        st.markdown('Face Detection')
        converted_img = np.array(image.convert('RGB'))
        image = cv2.cvtColor(converted_img,1)
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        st.sidebar.write("""
                 Scale Factor - A factor of 1.1 corresponds to an increase of 10%. Increase in the scale factor leads to increase in performance, as the number of detection passes is reduced. As a consequence the reliability by which a face is detected is reduced.
                 """)
                 
        scaleFactor = st.sidebar.slider("Scale Factor", 1.02,1.15,1.1,0.01)
        
        st.sidebar.write("""
                 Number of neighbors - determines the minimum number of neighbouring facial features that are needed to be present to indicate the detection of a face by the haar cascade classifier. Decrease in this factor increases the amount of false-positive detections.
                         """)
                 
        minNeighbors = st.sidebar.slider("Number of neighbors", 1, 15, 5, 1)
        
        st.sidebar.write("""
                Minimum Size - determines the minimum size of the detection window in pixels. Increase in the minimum detection window increases performance. As a consequence, smaller faces are going to be missed.
                 """)
                 
        minSize = st.sidebar.slider("Minimum Size", 10,50,20,1)
        
        faces_rect = haar_cascade.detectMultiScale(gray_scale, scaleFactor=scaleFactor,minNeighbors=minNeighbors,flags = cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces_rect:
            if w > minSize:
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 5)
        
        if len(faces_rect)>1:
            st.success("Found {} faces".format(len(faces_rect)))
        else:
            st.success("Found {} face".format(len(faces_rect)))

st.sidebar.title(' ') #Used to create some space between the filter widget and the comments section
st.sidebar.markdown(' ') #Used to create some space between the filter widget and the comments section
st.sidebar.subheader('Hope You Liked it!')