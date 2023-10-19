from loguru import logger
import glob 
import os
import numpy as np
import re
import pandas as pd
import skimage.io
from skimage.transform import resize
import h5py
import pyvips

from PIL import Image
import cv2
Image.MAX_IMAGE_PIXELS = None

from shapely.geometry import Point, Polygon
import json

logger.info("Starting JPGDataPreProcess/main.py")

#---->WSIPath used for glob

dataroot="/data/zrf/data/Data_jwy/P22062301/rawdata/"
saveroot="/data/zrf/data/processed_jwy_512"

window=512

train_test=True
check_polygons=True
train_ratio=0.2
np.random.seed(2023)


if __name__ == '__main__':         
    for img_path in glob.glob(os.path.join(dataroot,'*.tif')):
        # if img_path=='/data/zrf/data/Data_jwy/P22062301/rawdata/G3212069-D1-V11N29-105-L220623030.tif':
        #     continue
        
        # 图像名字
        imgname=os.path.basename(img_path)
        imgname=imgname.split('.')[0]
        outdir=os.path.join(saveroot,imgname)
        outdir_train=os.path.join(saveroot,imgname,'train')
        outdir_test=os.path.join(saveroot,imgname,'test')
        
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(outdir_train, exist_ok=True)
        os.makedirs(outdir_test, exist_ok=True)
        
        #load image
                
        image = Image.open(img_path)
        
        logger.info("Loading image: " + img_path)
        
        #load polygons in json file z作为勾画大轮廓
        with open(img_path.replace('.tif','.json') ,'r') as json_file:
            data = json.load(json_file)
        
        polygon_points=data['shapes'][0]['points']
        polygon = Polygon(polygon_points)
        
        # 获取大图像的宽度和高度
        large_width, large_height = image.size
        
        # 定义切分步长
        small_width, small_height = window, window
        step_x, step_y = small_width, small_height
        
        # 初始化起始坐标
        x_start, y_start = 0,0

        #---->Otsu threshold
        
        ostu_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        otsu_value, thresholded = cv2.threshold(ostu_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        logger.info('start croping')
        
        # 初始化小图像编号
        small_image_count = 0

        while y_start + small_height <= large_height:
            while x_start + small_width <= large_width:
                # 切分小图像
                small_image = image.crop((x_start, y_start, x_start + small_width, y_start + small_height))
                perc=np.average(thresholded[y_start:y_start+window,x_start:x_start+window])/255
                
                #更新起始坐标
                x_start += step_x
                
                #---->check wether in contour polygon
                if check_polygons and  not Point(x_start-small_width/2 , y_start+ small_height/2).within(polygon):
                    logger.warning('Patch'+ f'x:{x_start},y:{y_start}'+ " not in contour" )
                    continue
                
                #---->check image
                X = np.array(small_image)
                if not X.shape == (window, window, 3) :
                    logger.warning('Patch'+ f'x:{x_start},y:{y_start}'+ " not qualify(bcakground)" )
                    continue
                
                if perc > 0.7:
                    logger.warning('Patch'+ f'x:{x_start},y:{y_start}'+ " are bcakground" )
                    continue
                
                
                # 保存小图像
                if  train_test:
                    random = np.random.rand()
                    if random < train_ratio:
                        small_image.save( os.path.join(outdir_train,f"{x_start}_{y_start}.jpg") )
                    else:
                        small_image.save( os.path.join(outdir_test,f"{x_start}_{y_start}.jpg") )
                
                else:
                    small_image.save( os.path.join(outdir,f"{x_start}_{y_start}.jpg") )                
                
                # 和小图像编号
                small_image_count += 1

            # 重置x坐标，更新y坐标
            x_start = 0
            y_start += step_y

        # 关闭大图像
        image.close()


