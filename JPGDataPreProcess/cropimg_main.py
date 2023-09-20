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
from wsi_tile_cleanup import filters, utils
from PIL import Image
import cv2
Image.MAX_IMAGE_PIXELS = None


logger.info("Starting JPGDataPreProcess/main.py")

#---->WSIPath used for glob

dataroot="/data/zrf/data/Data_jwy/P22062301/rawdata/"
saveroot="/data/zrf/data/processed_jwy"

window=512

# def otsu_threshold(vi_wsi):
#     vi_wsi = resize(vi_wsi, (512, 512), mode='edge', anti_aliasing=True)
#     vi_wsi = pyvips.Image.new_from_array(vi_wsi)
#     return filters.otsu_threshold(vi_wsi)

# def background_perc(vi_bg, otsu):
#     vi_bg = pyvips.Image.new_from_array(vi_bg)
#     return filters.background_percent(vi_bg, otsu)

if __name__ == '__main__':         
    for img_path in glob.glob(os.path.join(dataroot,'*.tif')):
        
        # 图像名字
        imgname=os.path.basename(img_path)
        imgname=imgname.split('.')[0]
        outdir=os.path.join(saveroot,imgname)
        os.makedirs(outdir, exist_ok=True)
        
        #load image
                
        image = Image.open(img_path)
        
        logger.info("Loading image: " + img_path)
        
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

        # otsu = otsu_threshold(np.array(ostu_img))
        
        # cv2.imwrite('ostu.jpg',ostu_img)
        # cv2.imwrite('os.jpg',thresholded)

        
        logger.info('start croping')
        
        # 初始化小图像编号
        small_image_count = 0

        while y_start + small_height <= large_height:
            while x_start + small_width <= large_width:
                # 切分小图像
                small_image = image.crop((x_start, y_start, x_start + small_width, y_start + small_height))
                perc=np.average(thresholded[y_start:y_start+window,x_start:x_start+window])/255
                
                #logger
                # logger.info('Patch'+ f'x:{x_start},y:{y_start}'+ " are cropped" )
                
                #更新起始坐标
                x_start += step_x
                
                #---->check image
                X = np.array(small_image)
                if not X.shape == (window, window, 3) :
                    logger.warning('Patch'+ f'x:{x_start},y:{y_start}'+ " not qualify(bcakground)" )
                    continue
                
                if perc > 0.8:
                    logger.warning('Patch'+ f'x:{x_start},y:{y_start}'+ " are bcakground" )
                    continue
    
                # 保存小图像
                small_image.save( os.path.join(outdir,f"{x_start}_{y_start}_{small_image_count}.jpg") )
                
                # 和小图像编号
                small_image_count += 1

            # 重置x坐标，更新y坐标
            x_start = 0
            y_start += step_y

        # 关闭大图像
        image.close()


