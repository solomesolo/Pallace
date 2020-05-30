# after that we can import modules from directory above
import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import torch
import cv2

from config import MODEL_PATH, DEVICE
from CustomModel import PretrainedDensenet
from preprocess import preprocess_image


def get_model():
    model = PretrainedDensenet()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)
    model.eval()
    return model

model = get_model()

def get_prediction(file_path):
    image = cv2.imread(file_path)
    image_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model.forward(image_tensor.to(DEVICE))
    output = output.cpu().numpy()[0][0]
    print("output:", output)
    return output
    
    