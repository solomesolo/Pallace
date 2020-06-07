import cv2
import numpy as np

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
        print(self.conv)
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
    
    print("len cnts:", len(cnts))
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
    
        
    