import torch
from torchvision import models, transforms
import torch.nn as nn
import cv2
from PIL import Image
from sklearn import svm
import joblib
import warnings
warnings.filterwarnings('ignore')



#Load CNN Model
def effNetb2(device):
    model = models.efficientnet_b2(pretrained=False).to(device)
    
    in_features = 1024

    model._fc = nn.Sequential(
        nn.BatchNorm1d(num_features=in_features),    
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=256),
        nn.Dropout(0.4),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=128),
        nn.Linear(128, 2),).to(device)

    model.load_state_dict(torch.load('./Weights/effnet_b2.h5' , map_location=torch.device('cpu')) )
    model.eval()

    return model

#Normalising and transformation
def normalise_transform(img):
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])

  data_transform = transforms.Compose([
                  transforms.Resize((224, 224)),
                  transforms.ToTensor(),
                  normalize
              ])
  
  return data_transform(img)

def cv2_to_pil(img):
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_pil = Image.fromarray(img_rgb)
  return img_pil

#Calculating Prediction
def Predict(img , Mod):
    allClasses = ['Female', 'Male']
    out = Mod(img)
    _, predicted = torch.max(out.data, 1)
    allClasses.sort()
    labelPred = allClasses[predicted]
    return labelPred

#Getting Cropped Lips
def getting_Lips(img, model_yolo):
  results = model_yolo(img)

  detections = results.pandas().xyxy[0]

  if detections.empty:
      return
  else:
      bboxes = detections[['xmin', 'ymin', 'xmax', 'ymax']]

      for index, row in bboxes.iterrows():
          x1, y1, x2, y2 = row
          cropped_img = img[int(y1):int(y2), int(x1):int(x2)]

          return cropped_img

def loading_svm(lst):
  model = joblib.load('./Weights/svm_weights.joblib')
  y_pred = model.predict(lst)
  return y_pred

#Extracting Resnet Features
def get_resnet_features(image):
    with torch.no_grad():
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model.eval()
        features = model(image)
        features = torch.flatten(features, start_dim=1)
        array = features.numpy()
    pred = loading_svm(array.tolist())
    gender = 'Male' if pred[0] == 1 else 'Female'
    return gender