import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import pandas as pd
import random

from config import INPUT_SIZE
from web_app.config import INPUT_SIZE as PREDICT_INPUT_SIZE, DEVICE

import threading
from tqdm import tqdm
tqdm.pandas()


train_transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
#             transforms.RandomCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(30, translate=None, scale=(0.95,1.3), resample=False, fillcolor=0), # 30 is rotation
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

val_transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class MuraDataset(torch.utils.data.Dataset):
    def __init__(self, df, is_train=False):
        if is_train:
            self.transform = train_transform
        else:
            self.transform = val_transform
        
        # SAMPLING FOR DEBUG
#         df = df.sample(int(df.shape[0]/40))
        
        
        self.df=df
        self.to_augment = True
        
#         print("df.head():", df.head())
        
        # read images to memory
        self.inputs = []
        self.labels = []
        df.progress_apply(self._load_image_from_df_single, axis=1)
        self.shuffle_dataset = self.__on_epoch_end
        
        print("Few images with shape:", np.asarray(self.inputs[0]).shape)
        for i in range(3):
            plt.imshow(np.asarray(self.inputs[i]))
            plt.show()
        
        # multiprocessing
        
#         n_cores = 6
#         self.n_cores = n_cores
#         df_split = np.array_split(df, n_cores)
#         self.list_thread_outputs = {
#             'inputs': [[] for i in range(n_cores)],
#             'labels': [[] for i in range(n_cores)]
#         }
#         list_outputs = 0
#         p = Pool(n_cores)
#         list_outputs = p.map(apply_wrapper, df_split)
        
#         print("len(list_outputs):", len(list_outputs))
#         list_outputs_new = []
#         for list_part in list_outputs:
#             print(type(list_part), list_part)
#             list_outputs_new.extend(list_part.to_list())
        
        # using swifter
#         df.swifter.apply(self._load_image_from_df, axis=1)
        # multithreading
#         self._run_threads(df_split, self._load_image_from_df)
#         for i in range(n_cores):
#             print("len inputs i={} : {}".format(i, len(self.list_thread_outputs['inputs'][i])))
#             print("len labels i={} : {}".format(i, len(self.list_thread_outputs['labels'][i])))
            
            
#         for i in range(n_cores):
#             self.inputs.extend(self.list_thread_outputs['inputs'][0])
#             self.labels.extend(self.list_thread_outputs['labels'][0])
#             del self.list_thread_outputs['inputs'][0]
#             del self.list_thread_outputs['labels'][0]
        print("Loaded {} images and {} labels for the current dataset".format(str(len(self.inputs)), str(len(self.labels))))
        
    def __on_epoch_end(self, shuffle=True):
        if shuffle:
            c = list(zip(self.inputs, self.labels))
            del self.inputs
            del self.labels
            random.shuffle(c)
            self.inputs, self.labels = zip(*c)
            del c
            print("Dataset was shuffled")
    
    def _load_image_from_df_single(self, row):
        try:
            # some images from Pallace dataset downloaded from LabelBox with errors 
            img = cv2.imread(row['FilePath'], 1)
#         img = Preprocessing_single14(img)        
            img = Preprocessing(img)        
            img = Image.fromarray(img)

            label = row['Label']
            self.inputs.append(img)
            self.labels.append(label)    
        except:
            print("row:", row, "type(img):", type(img))
            print("row['FilePath']:", row['FilePath'])
        
    
#     def _apply_wrapper(self, i_worker, df_part):
#         df_part.apply((lambda row: self._load_image_from_df(i_worker, row)), axis=1)
    
#     def _run_threads(self, df_split, apply_func):
#         threads = []
#         for i_worker, df_part in enumerate(df_split):
#             threads.append(threading.Thread(target=self._apply_wrapper,args=(i_worker, df_part)))
#         [thread.start() for thread in threads]                                       
#         [thread.join() for thread in threads]

#     def _load_image_from_df(self, i_worker, row):
# #         print("row:", row)
#         img = cv2.imread(row['FilePath'], 1)

#         img = Preprocessing(img)        
#         img = Image.fromarray(img)

#         label = row['Label']
# #         self.inputs.append(img)
# #         self.labels.append(label)
#         # Threading writing
#         self.list_thread_outputs['inputs'][i_worker].append(img)
#         self.l￼ist_thread_outputs['labels'][i_worker].append(label)
        
#         len_list_worker = len(self.list_thread_outputs['inputs'][i_worker])
#         if i_worker == 0 and len_list_worker % 1000 == 0:
#             print("len of {} worker:{}. whole len: {}/{}".format(i_worker, len_list_worker, len_list_worker*self.n_cores, self.df.shape[0]))
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img = self.inputs[idx]
        label = self.labels[idx]
        
        image=np.zeros((img.size[0],img.size[1],3),dtype=np.uint8)
        for i in range(3):
            image[:,:,i]= np.asarray(img)
        img = Image.fromarray(image)
            
            
        if self.to_augment and self.transform:
            img=self.transform(img)
        else:
            transform_no_distr = transforms.Compose([
                transforms.RandomCrop(INPUT_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(30, translate=None, scale=(0.95,1.3), resample=False, fillcolor=0),
                transforms.ColorJitter(brightness=0.2),
                transforms.ToTensor()])
            img = transform_no_distr(img)
            
        label = torch.from_numpy(np.asarray(label)).double().type(torch.FloatTensor)
        return img, label
    

    
def read_preprocess_image(path):
    # read the image
    image = cv2.imread(path, 1)
    # preprocess image (crop to RoI and scale)
    img = Preprocessing(image, input_size=PREDICT_INPUT_SIZE)        

    # RGB from Grayscale image
    image = np.zeros((img.shape[0], img.shape[1], 3),dtype=np.uint8)
    for i in range(3):
        image[:,:,i]= np.asarray(img)
    
    # transform image to put into network
    image_pil = Image.fromarray(image)
    predict_transform = transforms.Compose([
            transforms.Resize(PREDICT_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    image_tensor = predict_transform(image_pil).to(DEVICE)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    
    return image, image_tensor
    

from skimage import filters
from skimage.measure import label,regionprops

def Preprocessing(image, input_size=INPUT_SIZE):
    if len(image.shape)==2:
        pass
    else:
        pass
        image=image[:,:,0]
    original_image_gray = np.copy(image)
    
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

    image = original_image_gray[MaximumBbox[0]:MaximumBbox[2],MaximumBbox[1]:MaximumBbox[3]]

    Longer,Shorter=(image.shape[0],image.shape[1]) if image.shape[0]>=image.shape[1] else (image.shape[1],image.shape[0])
    Start=int((Longer-Shorter)/2)
    imageR=np.zeros((Longer,Longer),np.uint8)
    if image.shape[0]>=image.shape[1]:
        imageR[:,Start:Start+Shorter]=image
    else:
        imageR[Start:Start+Shorter,:]=image
    image=imageR
        
#     image3=np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
#     for i in range(3):
#         image3[:,:,i]=image
    image3 = image
        

    image4=255-image3
    image3=Image.fromarray(image3)
    image3=image3.resize((input_size,input_size))
#     image4=Image.fromarray(image4)
#     image4=image4.resize((input_size,input_size))
    
    return np.asarray(image3)#,np.asarray(image4)]




def Preprocessing_single14(image):
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
        
    image = image[0,0,:,:]
#     plt.imshow(image)
#     plt.show()
    return Image.fromarray(image)
