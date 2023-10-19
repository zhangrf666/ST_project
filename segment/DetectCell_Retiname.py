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

import math
import sys
import time
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

num_classes = 2  # 1 class (person) + background
initial_lr=1e-3
num_epochs=3000
batch_size = 9

def transform(image, target):
    # image = F.to_tensor(image)
    # target = dict({'labels': torch.tensor(target)})
    return image, target

def evaluate_model(model, data_loader):
    model.eval()
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image for image in images)
            targets = [{k: v for k, v in target.items()} for target in targets]
            images, targets = list(zip(*[(image, target) for image, target in zip(images, targets)]))
            predictions = model(images)
            all_predictions.extend(predictions)
            all_ground_truths.extend(targets)

    return all_predictions, all_ground_truths



def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # data_transform = {
    #     "train": transforms.Compose([transforms.ToTensor(),
    #                                  transforms.RandomHorizontalFlip(0.5)]),
    #     "val": transforms.Compose([transforms.ToTensor()])
    # }

    # load train data set
    train_datapath='/data/zrf/data/processed_jwy_512/*/train'
    
    train_dataset=CellNucleusDataset(glob.glob(os.path.join(train_datapath,'*.json')))

    train_sampler = None

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减小训练时所需GPU显存，默认使用
    # if args.aspect_ratio_group_factor >= 0:
    #     train_sampler = torch.utils.data.RandomSampler(train_dataset)
    #     # 统计所有图像高宽比例在bins区间中的位置索引
    #     group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
    #     # 每个batch图片从同一高宽比例区间中取
    #     train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    
    train_batch_sampler=None
    if train_sampler:
        # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=0,
                                                        drop_last=True,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=0,
                                                        collate_fn=train_dataset.collate_fn)

    # load validation data set
    
    val_datapath='/data/zrf/data/processed_jwy_512/OS3203246-C1-V11N29-105-L220623029/test'
    
    val_dataset=CellNucleusDataset(glob.glob(os.path.join(val_datapath,'*.jpg')))
    val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=0,
                                                      collate_fn=val_dataset.collate_fn)

    # create model num_classes equal background + 20 classes
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained_backbone=True,num_classes=num_classes)
    
    #ssd300_vgg16
    # model = torchvision.models.detection.ssd300_vgg16(pretrained_backbone=True,num_classes=num_classes)

    

    #------> 训练模型
    model.to(device)

    # define optimizer
    # Set up the optimizer and loss function
    # params = [p for p in model.parameters() if p.requires_grad]
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    optimizer = torch.optim.SGD([
                                {'params': model.backbone.parameters(), 'lr': initial_lr*0.1},  # 调整骨干网络的学习率
                                {'params': model.head.parameters(), 'lr': initial_lr}], # 分类头部的学习率], # 回归头部的学习率, 默认使用的是1e-3
                                lr=initial_lr, momentum=0.9, weight_decay=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2600,2900], gamma=0.1)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None




    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []
    best_loss=10.0

    for epoch in range(args.start_epoch, num_epochs):
        
        # train for one epoch, printing every 10 iterations
        print('now start to train')
        loss_sum=[]
        if epoch==40: # 40 epoch后解冻骨干网络
            print('unfreeze backbone')
            for param in model.backbone.parameters():
                param.requires_grad = True
        
        for images, targets in train_data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # images, targets = list(zip(*[(image.to(device), target) for image, target in zip(images, targets)]))
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if not math.isfinite(losses):  # 当计算的损失为无穷大时停止训练
                print("Loss is {}, stopping training".format(losses))
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            loss_sum.append(losses.item())
        
        lr_scheduler.step()
        print('epoch:{} loss:{}'.format(epoch,np.mean(loss_sum)))
        
        # evaluate on the test dataset
        # print('now start to evaluate')
        # print('todo')
        
        #save best
        
        # save best model
        if np.mean(loss_sum)<best_loss and epoch>2000:
            best_loss=np.mean(loss_sum)
            best_model_wts = model.state_dict()

        if epoch % 50 == 0 and epoch>2000 or epoch==num_epochs-1:
            save_files = {
                'model': best_model_wts,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            if args.amp:
                save_files["scaler"] = scaler.state_dict()
            torch.save(save_files, os.path.join(  args.output_dir  ,"retinanet-epoch{}-loss{}.pth".format(epoch,best_loss) ))

                
            # np.savez('/data1/zrf/zhikong20230825/CompareModels/weight/mask_rcnn-model-CP-{}.npz'.format(epoch),tps=tps,socres=socres,n_gts=n_gts )
    # plot loss and lr curve
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # # 训练数据集的根目录(VOCdevkit)

    # 文件保存地址
    parser.add_argument('--output-dir', default='/data/zrf/STproj/ProjST/CELLSegment/output', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    # 学习率
    parser.add_argument('--lr', default=1e-4 , type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练的batch size
    parser.add_argument('--batch_size', default=16, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, help="Use torch.cuda.amp for mixed precision training")
    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
