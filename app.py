import streamlit as st
from PIL import Image
import torch
import helper
import numpy as np
from torchvision import transforms
import cv2
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="LipNet", page_icon=":lips:", layout="wide")

st.header("LipsNet")

st.sidebar.subheader(("Input a picture of a Face."))
file_up = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

#Selecting Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Load Yolo
model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path='./Weights/best.pt')


if file_up is not None:
  file_bytes = np.asarray(bytearray(file_up.read()), dtype=np.uint8)
  img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
  
  st.sidebar.image(img,caption = 'Uploaded Image.', width=None, use_column_width=None)
  st.write("")
  st.write("Just a second ...")
  
  cropped_img = helper.getting_Lips(img, model_yolo)
  st.image(cropped_img, caption = 'Extracted Lips', width=None, use_column_width=None)
  
  img1 = torch.reshape(helper.normalise_transform(helper.cv2_to_pil(cropped_img)) , (1, 3, 224, 224))

  col1, col2 = st.columns(2)
  with col1:
    st.header("EfficientNet-B2")
    model_cnn = helper.effNetb2(device)
    lbl1 = helper.Predict(img1, model_cnn)
    st.write("Prediction: ",lbl1)

  with col2:
    st.header("SVM")
    lbl2 = helper.get_resnet_features(img1)
    st.write("Prediction: ",lbl2)
