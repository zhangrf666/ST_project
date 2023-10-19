import os
import numpy as np
import torch
from PIL import Image 
import json
import sys
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms

class CellNucleusDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset=dataset
        self.label_dict={
            'cell':1,
            '‘':1
        }
        self.transform = transforms.Compose([                                       
                               transforms.ToTensor(),
                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        # load all image files, sorting them to
        # ensure that they are aligned
       

    def __getitem__(self, idx):
        # load images and masks
        json_path = self.dataset[idx]
        img_path = json_path.replace('.json','.jpg')
        
        
        nw, nh=512,512
        img = Image.open(img_path).convert('RGB')
        iw, ih = img.size
        img = img.resize((nw, nh))

        target = {}
        f=open(json_path)
        info= json.load(f)
        res=info['shapes']
        masks=[]
        obj_ids=[]
        num_objs = len(res)
        for i in range(num_objs):
            masks.append(res[i]['points'])
            obj_ids.append(res[i]['label'])


        # get bounding box coordinates for each mask
        
        boxes = []
        labels=[]
        for i in range(num_objs):
            pos = np.array(masks[i])
            xmin = np.min(pos[:,0]) *((nw / iw))
            xmax = np.max(pos[:,0]) *((nw / iw))
            ymin = np.min(pos[:,1]) *((nh / ih))
            ymax = np.max(pos[:,1]) *((nh / ih))
            boxes.append([xmin, ymin, xmax, ymax])

            labels.append(self.label_dict[obj_ids[i]])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels=torch.as_tensor(labels)


        image_id = torch.tensor([idx+1])
        # area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        # target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return np.shape(self.dataset)[0]

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


    def get_height_and_width(self, idx):
        json_path = self.dataset[idx][1]
        f=open(json_path)
        info= json.load(f)
        res=info['shapes']

        data_height = int(info['imageHeight'])
        data_width = int(info['imageWidth'])
        return data_height, data_width


if __name__=='__main__':
    normal_dataset=np.load('/data/zrf/zhikongcode/faster_rcnn/dataloaders/CRL_imgpath.npz')['val_files']
    data=CellNucleusDataset(normal_dataset,None)
    for i in tqdm(range(len(data))):
        try:
            a=data[i]
        except Exception as e: 
            print(i,':')
            print(e)
            print()
            sys.exit(1)

        
    
    