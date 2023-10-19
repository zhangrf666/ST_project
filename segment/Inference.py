
import os
import datetime
import numpy as np
import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from utils.Labelme_Dataset import CellNucleusDataset
import torchvision.transforms as T
import torch.nn as nn
import glob
import h5py
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
from torchvision.ops import nms


model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained_backbone=True,num_classes=2)
# 加载训练好的RetinaNet模型

weights_path='/data/zrf/STproj/ProjST/CELLSegment/output/retinanet-epoch2999-loss0.037208300083875656.pth'
weights_dict = torch.load(weights_path, map_location='cpu')
weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
model.load_state_dict(weights_dict)
model.eval()

# 定义数据变换

def transform(image):
    image = F.to_tensor(image)
    image= F.normalize(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return [image]

# 定义一个函数来可视化检测结果，并应用NMS
def visualize_detection(image, detections, threshold=0.5, iou_threshold=0.5):
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    # 获取非极大值抑制后的边界框索引
    keep = nms(detections["boxes"], detections["scores"], iou_threshold)

    for idx in keep:
        score = detections["scores"][idx]
        label = detections["labels"][idx]
        box = detections["boxes"][idx].tolist()

        if score >= threshold:
            box = [round(i, 2) for i in box]  # 四舍五入
            draw.rectangle(box, outline="red", width=3)
            # draw.text((box[0], box[1]), f"Label: {label}, Score: {score:.2f}", fill="red")

    return image

# 加载图像
from tqdm import tqdm   

imgpaths=glob.glob('/data/zrf/data/processed_jwy_512/OS3203246-C1-V11N29-105-L220623029/test/*.jpg')
output_dir='/data/zrf/data/processed_jwy_512/OS3203246-C1-V11N29-105-L220623029/test_result'
for img_path in tqdm(imgpaths):
    image = Image.open(img_path)
    #'/data/zrf/data/processed_jwy_512/OS3203246-C1-V11N29-105-L220623029/test/4608_5120.jpg'
    images = transform(image)

    # 进行推断
    with torch.no_grad():
        predictions = model(images)

    # 可视化检测结果
    image_with_detections = visualize_detection(image, predictions[0], threshold=0.1)

    # 显示图像
    # image_with_detections.show()

    image_with_detections.save(os.path.join(output_dir,os.path.basename(img_path)))