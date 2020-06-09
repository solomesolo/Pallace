import cv2
import numpy as np
<<<<<<< HEAD
import torch
import torch.nn.functional as F
from torchvision import transforms
import argparse
from PIL import Image
# from common import config

Trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

Normal = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

invTrans = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.], std=[1/0.229, 1/0.224, 1/0.225]),
                               transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
                               ])


def generate_grad_cam(net, ori_image):
    """
    :param net: deep learning network(ResNet DataParallel object)
    :param ori_image: the original image
    :return: gradient class activation map
    """
    input_image = Trans(ori_image)

    feature = None
    gradient = None

    def func_f(module, input, output):
        nonlocal feature
        feature = output.data.cpu().numpy()

    def func_b(module, grad_in, grad_out):
        nonlocal gradient
        gradient = grad_out[0].data.cpu().numpy()

    net.features.layer4.register_forward_hook(func_f)
    net.features.layer4.register_backward_hook(func_b)
#     net.module.layer4.register_backward_hook(func_b)

    out = net(input_image.unsqueeze(0))

    pred = (out.data > 0.5)

    net.zero_grad()

    loss = F.binary_cross_entropy(out, pred.float())
    loss.backward()

    feature = np.squeeze(feature, axis=0)
    gradient = np.squeeze(gradient, axis=0)

    weights = np.mean(gradient, axis=(1, 2), keepdims=True)

    cam = np.sum(weights * feature, axis=0)

    cam = cv2.resize(cam, (224, 224))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    cam = 1.0 - cam
    cam = np.uint8(cam * 255)

    return cam


def localize(cam_feature, ori_image):
    """
    localize the abnormality region using grad_cam feature
    :param cam_feature: cam_feature by generate_grad_cam
    :param ori_image: the original image
    :return: img with heatmap, the abnormality region is highlighted
    """
    ori_image = np.array(ori_image)
    activation_heatmap = cv2.applyColorMap(cam_feature, cv2.COLORMAP_JET)
    activation_heatmap = cv2.resize(activation_heatmap, (ori_image.shape[1], ori_image.shape[0]))
    img_with_heatmap = 0.15 * np.float32(activation_heatmap) + 0.85 * np.float32(ori_image)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap) * 255
    return img_with_heatmap


def localize2(cam_feature, ori_image):
    """
    localize the abnormality region using grad_cam feature
    :param cam_feature: cam_feature by generate_grad_cam
    :param ori_image: input of the network
    :return: img with heatmap, the abnormality region is in a red window
    """
    ori_image = np.array(ori_image)
    cam_feature = cv2.resize(cam_feature, (ori_image.shape[1], ori_image.shape[0]))
    crop = np.uint8(cam_feature > 0.7 * 255)
    h = ori_image.shape[0]
    w = ori_image.shape[1]
    ret, markers = cv2.connectedComponents(crop)
    branch_size = np.zeros(ret)
    for i in range(h):
        for j in range(w):
            t = int(markers[i][j])
            branch_size[t] += 1
    branch_size[0] = 0
    max_branch = np.argmax(branch_size)
    mini = h
    minj = w
    maxi = -1
    maxj = -1
    for i in range(h):
        for j in range(w):
            if markers[i][j] == max_branch:
                if i < mini:
                    mini = i
                if i > maxi:
                    maxi = i
                if j < minj:
                    minj = j
                if j > maxj:
                    maxj = j
    img_with_window = np.uint8(ori_image)
    img_with_window[mini:mini+2, minj:maxj, 0:1] = 255
    img_with_window[mini:mini+2, minj:maxj, 1:3] = 0
    img_with_window[maxi-2:maxi, minj:maxj, 0:1] = 255
    img_with_window[maxi-2:maxi, minj:maxj, 1:3] = 0
    img_with_window[mini:maxi, minj:minj+2, 0:1] = 255
    img_with_window[mini:maxi, minj:minj+2, 1:3] = 0
    img_with_window[mini:maxi, maxj-2:maxj, 0:1] = 255
    img_with_window[mini:maxi, maxj-2:maxj, 1:3] = 0

    return img_with_window


def generate_local(cam_features, inputs):
    """
    :param cam_features: numpy array of shape = (B, 224, 224), pixel value range [0, 255]
    :param inputs: tensor of size = (B, 3, 224, 224), with mean and std as Imagenet
    :return: local image
    """
    b = cam_features.shape[0]
    local_out = []
    for k in range(b):
        ori_img = invTrans(inputs[k]).cpu().numpy()
        ori_img = np.transpose(ori_img, (1, 2, 0))
        ori_img = np.uint8(ori_img * 255)

        crop = np.uint8(cam_features[k] > 0.7)
        ret, markers = cv2.connectedComponents(crop)
        branch_size = np.zeros(ret)
        h = 224
        w = 224
        for i in range(h):
            for j in range(w):
                t = int(markers[i][j])
                branch_size[t] += 1
        branch_size[0] = 0
        max_branch = np.argmax(branch_size)
        mini = h
        minj = w
        maxi = -1
        maxj = -1
        for i in range(h):
            for j in range(w):
                if markers[i][j] == max_branch:
                    if i < mini:
                        mini = i
                    if i > maxi:
                        maxi = i
                    if j < minj:
                        minj = j
                    if j > maxj:
                        maxj = j
        local_img = ori_img[mini: maxi + 1, minj: maxj + 1, :]
        local_img = cv2.resize(local_img, (224, 224))
        local_img = Image.fromarray(local_img)
        local_img = Normal(local_img)
        local_out += [local_img]
    local_out = torch.stack(local_out)
    return local_out


# if __name__ == '__main__':
    
def locate(net, ori_image_pil):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-m', '--model_path',
#                         default='models/resnet50_b16.pth.tar', type=str, required=False, help='filepath of the model')
#     parser.add_argument('--img_path', type=str, required=False, help='filepath of query input')
#     args = parser.parse_args()

#     net = torch.load(args.model_path)['net']

#     ori_image = Image.open(args.img_path).convert('RGB')
    ori_image = ori_image_pil
    cam_feature = generate_grad_cam(net, ori_image)
    result1 = localize(cam_feature, ori_image)
    result2 = localize2(cam_feature, ori_image)
    result2 = Image.fromarray(result2)
    
    return result1, result2
#     cv2.imwrite(args.img_path[:-4] + "_m.png", result1)
#     result2.save(args.img_path[:-4] + "_w.png")
# ------------------------------------------------------------------------------------------------------------------------



























# ------------------------------------------------------------------------------------------------------------------------
# from misc_functions import get_example_params, save_class_activation_images


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, orig_x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(orig_x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier
#         x = self.model.classifier(x)
        x = self.model(orig_x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.features.zero_grad()
#         self.model.classifier.zero_grad()
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam


# if __name__ == '__main__':
#     # Get params
#     target_example = 0  # Snake
#     (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
#         get_example_params(target_example)
# Grad cam
prep_img = image
grad_cam = GradCam(model, target_layer=11)
# Generate cam mask
cam = grad_cam.generate_cam(image.to('cuda'), 0)
# Save mask
# save_class_activation_images(original_image, cam, file_name_to_export)
print('Grad cam completed')
# ------------------------------------------------------------------------------------------------------------------------


















# ------------------------------------------------------------------------------------------------------------------------
from grad_cam import *

# Can work with any model, but it assumes that the model has a
# feature method, and a classifier method,
# as in the VGG models in torchvision.
# model = models.resnet50(pretrained=True)
img = np.transpose(images[0].numpy(), [1,2,0])
print("img shape:", img.shape)
feature_module = list(model.features.modules())[-50]#(by_name='denselayer32')
feature_module = list(model.features.modules())[2]
use_cuda = True

# model_2.layer[0].weight

grad_cam = GradCam(model=model, feature_module=feature_module, \
                   target_layer_names=["densenet32"], use_cuda=use_cuda)

# img = cv2.imread(args.image_path, 1)
# img = np.float32(cv2.resize(img, (224, 224))) / 255
input = preprocess_image(img)
print("input:", input.shape)

# If None, returns the map for the highest scoring category.
# Otherwise, targets the requested index.
target_index = None
mask = grad_cam(input, target_index)

show_cam_on_image(img, mask)

gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
print(model._modules.items())
gb = gb_model(input, index=target_index)
gb = gb.transpose((1, 2, 0))
cam_mask = cv2.merge([mask, mask, mask])
cam_gb = deprocess_image(cam_mask*gb)
gb = deprocess_image(gb)

cv2.imwrite('gb.jpg', gb)
cv2.imwrite('cam_gb.jpg', cam_gb)
=======

import torch
import torch.nn as nn
import torch.nn.functional as F



class CAM(nn.Module):
    def __init__(self, model_to_convert, get_fc_layer=lambda m: m.fc1,score_fn=F.sigmoid, resize=True):
        super().__init__()
#         self.backbone = nn.Sequential(*list(model_to_convert.children())[:-1])
        self.backbone = model_to_convert.features
        self.fc = get_fc_layer(model_to_convert)
        self.conv  =  nn.Conv2d(self.fc.in_features, self.fc.out_features, kernel_size=1)
        self.conv.weight = nn.Parameter(self.fc.weight.data.unsqueeze(-1).unsqueeze(-1))
        self.conv.bias = self.fc.bias
        self.score_fn = score_fn
        self.resize = resize
        self.eval()
        
    def forward(self, x, out_size=None):
        with torch.no_grad():
            batch_size, c, *size = x.size()
            feat = self.backbone(x)
            cmap = self.score_fn(self.conv(feat))
            if self.resize:
                if out_size is None:
                    out_size = size
                cmap = F.upsample(cmap, size=out_size, mode='bilinear')
            pooled = F.adaptive_avg_pool2d(feat,output_size=1)
            flatten = pooled.view(batch_size, -1)
            cls_score = self.score_fn(self.fc(flatten))
            weighted_cmap =  (cmap*cls_score.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
    #         return cmap, cls_score, weighted_cmap
            heat_map = weighted_cmap.data.cpu().numpy()[0]
            return cls_score, heat_map  # cls_score, here for debug only. Remove then
    
    
    
import matplotlib.pyplot as plt
    
def get_bb_from_heatmap(heat_map, thr_value=None, mean_value_mul=None):
    heat_map = (heat_map*255).astype('uint8')
    
    if thr_value is None:
#         if heat_map.mean()
        thr_value = heat_map.mean()
    if mean_value_mul is not None:
        thr_value = heat_map.mean()*mean_value_mul
    thresh = cv2.threshold(heat_map, thr_value, 255, cv2.THRESH_BINARY)[1]
    
    # Find contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    list_boxes = []
    max_box_area = 0
    max_box_idx = 0
    for idx, c in enumerate(cnts):
        x,y,w,h = cv2.boundingRect(c)
        list_boxes.append([x,y,w,h])
        if w*h >  max_box_area:
            max_box_area = w*h
            max_box_idx = idx

#         cv2.rectangle(image_np, (x, y), (x + w, y + h), (36,255,12), 2)
#         cv2.rectangle(heat_map_draw, (x, y), (x + w, y + h), (36,255,12), 2)
#         plt.imshow(image_np)
#         plt.imshow(heat_map_draw)
#         plt.show()
        
        
#         print("x: {}, y:{}, w:{}, h:{}".format(x,y,w,h))
#     plt.imshow(thresh)
#     plt.show()

    if list_boxes:
        bounding_box = list_boxes[max_box_idx]
    else:
        bounding_box = [0,0,10,10]
        
    return bounding_box
    
        
    
>>>>>>> first-stage
