
import torch
from torchvision import models
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

import cv2
from skimage.measure import label,regionprops
from skimage import filters

import numpy as np
from PIL import Image

from collections import OrderedDict

# --------------------------- MODEL --------------------------- 

def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    return model

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
        
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))
        
class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=out.shape[2], stride=1).view(features.size(0), -1)
        #out = F.max_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out
model_ft1 = densenet169(pretrained=False)
num_features = 1664
model_ft1.classifier = nn.Linear(num_features, 2)
best_single_model = model_ft1


import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    
    
best_single_model.load_state_dict(torch.load('best_single/params.pkl'))#.load('../params.pkl')#('./WeightUnifiedDL'+str(Window)+'.pkl')
best_single_model.to('cuda')
best_single_model.eval()


# --------------------------- DATA --------------------------- 
Window=320
Mean=[0.485, 0.456, 0.406]

def Preprocessing(Image):
    if len(Image.shape)==2:
        pass
    else:
        Image=Image[:,:,0]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    Image = clahe.apply(Image)
    
    otsu=filters.threshold_otsu(Image)
    Seg=np.zeros(Image.shape)
    Seg[Image>=0.5*otsu]=255
    Seg=Seg.astype(np.int)
    
    ConnectMap=label(Seg, connectivity= 2)
    Props = regionprops(ConnectMap)
    Area=np.zeros([len(Props)])
    Area=[]
    Bbox=[]
    for j in range(len(Props)):
        Area.append(Props[j]['area'])
        Bbox.append(Props[j]['bbox'])
    Area=np.array(Area)
    Bbox=np.array(Bbox)
    argsort=np.argsort(Area)
    Area=Area[argsort]
    Bbox=Bbox[argsort]
    Area=Area[::-1]
    Bbox=Bbox[::-1,:]
    MaximumBbox=Bbox[0]

    Image=Image[MaximumBbox[0]:MaximumBbox[2],MaximumBbox[1]:MaximumBbox[3]]
    Longer,Shorter=(Image.shape[0],Image.shape[1]) if Image.shape[0]>=Image.shape[1] else (Image.shape[1],Image.shape[0])
    Start=int((Longer-Shorter)/2)
    ImageR=np.zeros((Longer,Longer),np.uint8)
    if Image.shape[0]>=Image.shape[1]:
        ImageR[:,Start:Start+Shorter]=Image
    else:
        ImageR[Start:Start+Shorter,:]=Image
    Image=ImageR
        
    Image3=np.zeros((Image.shape[0],Image.shape[1],3),dtype=np.uint8)
    for i in range(3):
        Image3[:,:,i]=Image

    Image4=255-Image3
    Image3=Img.fromarray(Image3)
    Image3=Image3.resize((Window,Window))
    #Image3.save('C:/Data/'+PatientPath[:-4]+'_PreProcCroppedL'+str(Window)+'.png')
    Image4=Img.fromarray(Image4)
    Image4=Image4.resize((Window,Window))
    #Image4.save('C:/Data/'+PatientPath[:-4]+'_PreProcCroppedD'+str(Window)+'.png')
    
    
    
    # CONCATE 
    TestImageL = np.asarray(Image3)
    TestImageD = np.asarray(Image4)
    [TestImageL,TestImageD]=Preprocessing(Image)
    TestImageL=np.expand_dims(TestImageL.transpose(2,0,1),axis=0)
    TestImageD=np.expand_dims(TestImageD.transpose(2,0,1),axis=0)
    Image=np.concatenate((TestImageL,TestImageD),axis=0)/255.0
    for j in range(3):
        Image[:,j,:,:]-=Mean[j]
    
    return Image


def Preprocessing(image):
    Mean=[0.485, 0.456, 0.406]
    
    if len(image.shape)==2:
        pass
    else:
        image=image[:,:,0]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    
    otsu=filters.threshold_otsu(image)
    Seg=np.zeros(image.shape)
    Seg[image>=0.5*otsu]=255
    Seg=Seg.astype(np.int)
    
    ConnectMap=label(Seg, connectivity= 2)
    Props = regionprops(ConnectMap)
    Area=np.zeros([len(Props)])
    Area=[]
    Bbox=[]
    for j in range(len(Props)):
        Area.append(Props[j]['area'])
        Bbox.append(Props[j]['bbox'])
    Area=np.array(Area)
    Bbox=np.array(Bbox)
    argsort=np.argsort(Area)
    Area=Area[argsort]
    Bbox=Bbox[argsort]
    Area=Area[::-1]
    Bbox=Bbox[::-1,:]
    MaximumBbox=Bbox[0]

    image=image[MaximumBbox[0]:MaximumBbox[2],MaximumBbox[1]:MaximumBbox[3]]
    Longer,Shorter=(image.shape[0],image.shape[1]) if image.shape[0]>=image.shape[1] else (image.shape[1],image.shape[0])
    Start=int((Longer-Shorter)/2)
    ImageR=np.zeros((Longer,Longer),np.uint8)
    if image.shape[0]>=image.shape[1]:
        ImageR[:,Start:Start+Shorter]=image
    else:
        ImageR[Start:Start+Shorter,:]=image
    image=ImageR
        
    Image3=np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    for i in range(3):
        Image3[:,:,i]=image
    

    Image4=255-Image3
    Image3=Image.fromarray(Image3)
    Image3=Image3.resize((Window,Window))
    Image4=Image.fromarray(Image4)
    Image4=Image4.resize((Window,Window))
    
    
    
    # CONCATE 
    TestImageL = np.asarray(Image3)
    TestImageD = np.asarray(Image4)
#     [TestImageL,TestImageD]=Preprocessing(image)
    TestImageL=np.expand_dims(TestImageL.transpose(2,0,1),axis=0)
    TestImageD=np.expand_dims(TestImageD.transpose(2,0,1),axis=0)
    image=np.concatenate((TestImageL,TestImageD),axis=0)/255.0
    for j in range(3):
        image[:,j,:,:]-=Mean[j]
        
#     image = image[0,0,:,:]
#     plt.imshow(image)
#     plt.show()
    return image


def predict_image(image):
    image = Preprocessing(image)
    
    X=torch.from_numpy(image)
    X=X.float()
    X=Variable(X).to('cuda')
    
    Pred = best_single_model.forward(X)
#     print("Finishing Pred")
    Pred=F.softmax(Pred).data.cpu().numpy()
    Pred=np.mean(Pred[:,1])
    return Pred
#     Preds.append(Pred)