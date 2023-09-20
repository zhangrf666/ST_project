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

'''
hf.create_dataset('images', shape=(spotData.shape[0],) + (window, window) + (3,), dtype=np.uint8)  #row image data
hf.create_dataset('points', shape=(spotData.shape[0],) + (2,), dtype=np.uint8)   # pos of spot x,y
hf.create_dataset('index', shape=(spotData.shape[0],) + (1,), dtype=str) #spot indexW

'''

logger.add("/data/zrf/STproj/ProjST/logs/PreprocessBreast.log", rotation="50 MB", compression="zip")
logger.info("Starting JPGDataPreProcess/main.py")

#---->WSIPath used for glob
metadataPath = "/data/zrf/data/Data_Breast/hist2tscript/metadata.csv"
dataroot="/data/zrf/data/Data_Breast/hist2tscript"
saveroot="/data/zrf/data/ProcessData_Breast/imgh5"

window=512

def otsu_threshold(vi_wsi):
    vi_wsi = resize(vi_wsi, (512, 512), mode='edge', anti_aliasing=True)
    vi_wsi = pyvips.Image.new_from_array(vi_wsi)
    return filters.otsu_threshold(vi_wsi)

def background_perc(vi_bg, otsu):
    vi_bg = pyvips.Image.new_from_array(vi_bg)
    return filters.background_percent(vi_bg, otsu)

if __name__ == '__main__':
    metadata=pd.read_csv(metadataPath,sep=',',header=0)
    for index, row_metadata in metadata.iterrows():
        
        img_path=os.path.join(dataroot,row_metadata['histology_image'])
        stdata_path=os.path.join(dataroot,row_metadata['count_matrix'])
        spot_path=os.path.join(dataroot,row_metadata['spot_coordinates'])
        
        #---->ST data and spot data
        
        stData=pd.read_csv(stdata_path,sep='\t',compression='gzip',header=0,index_col=0)
        spotData=pd.read_csv(spot_path,compression='gzip',header=0,index_col=0)
        image = skimage.io.imread(img_path)
        
        logger.info("Loading image: " + img_path)
        
        #----> output file       
        output_file= os.path.join(saveroot,row_metadata['patient']+'_'+row_metadata['replicate']+'_cropimage.h5')
        
        #---->Otsu threshold
        otsu = otsu_threshold(image)
       
        with h5py.File(output_file, 'w') as hf:
            
            hf.create_dataset('images', shape=(spotData.shape[0],) + (window, window) + (3,), dtype=np.uint8)  #row image data
            hf.create_dataset('points', shape=(spotData.shape[0],) + (2,), dtype=np.uint8)   # pos of spot x,y
            hf.create_dataset('index', shape=(spotData.shape[0],) + (1,), dtype=h5py.special_dtype(vlen=str))
        
            #---->Tile img according to ST data
            data = []
            for i,(index, row_spot) in enumerate(spotData.iterrows()):
                x = int(round(row_spot["X"]))
                y = int(round(row_spot["Y"]))
                try:
                    X = image[(y + (-window // 2)):(y + (window // 2)), (x + (-window // 2)):(x + (window // 2)), :]
                except:
                    logger.warning('Patch'+ index+ ' x,y : '+ str(int(row_spot["X"])) + str(int(row_spot["Y"])) + " fail to crop " )
                    continue
                
                # check the img  size and whether it is background                            
                if not X.shape == (window, window, 3) :
                    logger.warning('Patch'+ index+ ' x,y : '+ str(int(row_spot["X"])) + str(int(row_spot["Y"])) + " not qualify(bcakground)" )
                    continue
                
                perc=background_perc(X, otsu)
                if perc > 0.5:
                    logger.warning('Patch'+ index+ ' x,y : '+ str(int(row_spot["X"])) + str(int(row_spot["Y"])) + " are bcakground" )
                    continue
                
                #---->Save img and spot data
                hf['images'][i] = X
                hf['points'][i] = np.array([x, y])
                hf['index'][i] = index
                
            #---->store st data of each patient
            loaded_data=hf['index'][:]
            loaded_data = [str(item[0], 'utf-8') for item in loaded_data]            
            filtered_data = [index for index in loaded_data if index in stData.index]
            
            if not len(filtered_data) == len(loaded_data):
                logger.warning('Patient'+ row_metadata['patient']+ ' replicate : '+ row_metadata['replicate'] +
                               " ST data and spot data are not match: "+str((set(loaded_data)-set(filtered_data))) )
                
            
            stData_filtered=stData.loc[filtered_data]
            df_reordered= stData_filtered.reindex(filtered_data)
            df_reordered.to_csv(os.path.join(saveroot,row_metadata['patient']+'_'+row_metadata['replicate']+'_stdata.csv'),compression='gzip',header=True,index=True)
            
    
    
    logger.info("Finished JPGDataPreProcess/main.py")                

