import torch
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.transforms import functional as F
import glob
import torchvision
from torchvision.models.detection.retinanet import RetinaNet
from utils.Labelme_Dataset import CellNucleusDataset

import os

model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained_backbone=True,num_classes=2)
# 加载训练好的RetinaNet模型

weights_path='/data/zrf/STproj/ProjST/CELLSegment/output/retinanet-epoch2050-loss0.07256290440758069.pth'
weights_dict = torch.load(weights_path, map_location='cpu')
weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
model.load_state_dict(weights_dict)
model.eval()

# 加载测试数据集，使用与训练数据集相似的自定义 Dataset 和 DataLoader

val_datapath='/data/zrf/data/processed_jwy_512/G3212069-D1-V11N29-105-L220623030/val'

val_dataset=CellNucleusDataset(glob.glob(os.path.join(val_datapath,'*.json')))
val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=1,
                                                    shuffle=False,
                                                    pin_memory=True,
                                                    num_workers=0,
                                                    collate_fn=val_dataset.collate_fn)

# 创建一个函数来计算评估指标
def evaluate_model(model, data_loader):
    model.eval()
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image for image in images)
            targets = [{k: v for k, v in target.items()} for target in targets]
            # images, targets = list(zip(*[(image, target) for image, target in zip(images, targets)]))
            predictions = model(images)
            all_predictions.extend(predictions)
            all_ground_truths.extend(targets)

    return all_predictions, all_ground_truths


# 在测试数据集上进行评估
test_predictions, test_ground_truths = evaluate_model(model, val_data_set_loader)

from torch import tensor
from torchmetrics.detection import MeanAveragePrecision

metric = MeanAveragePrecision(iou_type="bbox",max_detection_thresholds=[50,100,500])
metric.update(test_predictions, test_ground_truths)
from pprint import pprint
pprint(metric.compute())