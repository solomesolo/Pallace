# after that we can import modules from directory above
import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# import io
import torchvision.transforms as transforms
from PIL import Image

from config import INPUT_SIZE
from datasets_gpu import *

# def preprocess_image(image):
#     image = Preprocessing(image)
#     image = Image.fromarray(image)
#     my_transforms = transforms.Compose([transforms.Resize(INPUT_SIZE),
#                                         transforms.ToTensor(),
#                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
# #     image = Image.open(io.BytesIO(image_bytes))
#     return my_transforms(image).unsqueeze(0)
