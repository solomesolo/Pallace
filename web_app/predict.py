# after that we can import modules from directory above
import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import torch
import cv2

<<<<<<< HEAD
from config import MODEL_PATH, DEVICE
from CustomModel import PretrainedDensenet
from preprocess import preprocess_image


def get_model():
    model = PretrainedDensenet()
    model.load_state_dict(torch.load(MODEL_PATH))
=======
from config import LIST_MODELS, DEVICE
from model import PretrainedDensenet
from data_utils import *
from localization import *



def get_model(model_path):
    model = PretrainedDensenet()
    model.load_state_dict(torch.load(model_path))
>>>>>>> first-stage
    model.to(DEVICE)
    model.eval()
    return model

<<<<<<< HEAD
model = get_model()

=======
# init models
list_cam_models = [CAM(get_model(model_path)) for model_path in LIST_MODELS]

# old (only prediction w/o heatmap)
>>>>>>> first-stage
def get_prediction(file_path):
    image = cv2.imread(file_path)
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model.forward(image_tensor.to(DEVICE))
    output = output.cpu().numpy()[0][0]
    print("output:", output)
    return output
<<<<<<< HEAD
    
    
=======

def get_prediction_and_heat_map(file_path):
    image, image_tensor = read_preprocess_image(file_path)
    
    # predict anomality and heat map of CAM (class activation map)
    list_scores = []
    list_heat_maps = []
    for cam in list_cam_models:   
        cls_score_tensor, heat_map = cam(image_tensor)
        list_scores.append(cls_score_tensor.cpu().numpy()[0][0])
        list_heat_maps.append(heat_map)
    
    cls_score = np.array(list_scores).mean()
    heat_map = np.mean(list_heat_maps, axis=0)
    
    bounding_box = get_bb_from_heatmap(heat_map, mean_value_mul=1)
    
    return image, cls_score, heat_map, bounding_box 
>>>>>>> first-stage
