# after that we can import modules from directory above
import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import torch
import cv2

from config import MODEL_PATH, DEVICE
from model import PretrainedDensenet
# from preprocess import preprocess_image
from data_utils import *
from localization import *



def get_model():
    model = PretrainedDensenet()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()
    return model

model = get_model()
cam = CAM(model)

def get_prediction(file_path):
    image = cv2.imread(file_path)
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model.forward(image_tensor.to(DEVICE))
    output = output.cpu().numpy()[0][0]
    print("output:", output)
    return output

def get_prediction_and_heat_map(file_path):
    image, image_tensor = read_preprocess_image(file_path)
#     image = cv2.imread(file_path, 1)
#     image_tensor = preprocess_image(image)
    
    # predict anomality and heat map of CAM (class activation map)
    cls_score, heat_map = cam(image_tensor)
    bounding_box = get_bb_from_heatmap(heat_map, mean_value_mul=1)
    
    return image, cls_score, heat_map, bounding_box # cls_score, heat_map
    
    