# import functions
import labelme2coco
# from labelme2coco import get_coco_from_labelme_folder, save_json

# set labelme training data directory
labelme_train_folder = "/data/zrf/data/processed_jwy_512/G3212069-D1-V11N29-105-L220623030/val"

# set labelme validation data directory
# labelme_val_folder = "tests/data/labelme_annot"

# set path for coco json to be saved
export_dir = "/data/zrf/data/processed_jwy_512/G3212069-D1-V11N29-105-L220623030/val_coco/"

# create train coco object
train_coco = labelme2coco.get_coco_from_labelme_folder(labelme_train_folder)

# export train coco json
labelme2coco.save_json(train_coco.json, export_dir+"val_coco.json")

# create val coco object
# val_coco = get_coco_from_labelme_folder(labelme_val_folder, coco_category_list=train_coco.json_categories)

# export val coco json
# save_json(val_coco.json, export_dir+"val.json")