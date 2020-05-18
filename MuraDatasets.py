import torch
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

class MuraDataset(torch.utils.data.Dataset):
    
    def __init__(self,df,transform=None):
        self.df=df
        self.transform=transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        img_name=self.df.iloc[idx,0]
        img = cv2.imread(img_name)#, -1) # 1 - read always in color
#         plt.imshow(img)
#         plt.show()
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if img.shape[2] != 3:
            print("img.shape:", img.shape)
        img = Image.fromarray(img)
        label=self.df.iloc[idx,1]

        if self.transform:
            img=self.transform(img)
        label = torch.from_numpy(np.asarray(label)).double().type(torch.FloatTensor)
        return img, label

