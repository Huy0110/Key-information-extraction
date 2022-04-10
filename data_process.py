import os
import shutil

import tqdm
from PIL import Image
import json
import random
from PIL import ImageDraw, ImageFont
from matplotlib.font_manager import json_load
folder_path_train = 'train/json'
folder_path_val = 'val/json'
folder_path_test = 'test/json'
folder_path_train_image = 'train/image'
folder_path_val_image = 'val/image'
folder_path_test_image = 'test/image'

if not os.path.exists(folder_path_train):
    os.makedirs(folder_path_train)
if not os.path.exists(folder_path_val):
    os.makedirs(folder_path_val)
if not os.path.exists(folder_path_test):
    os.makedirs(folder_path_test)
if not os.path.exists(folder_path_train_image):
    os.makedirs(folder_path_train_image)
if not os.path.exists(folder_path_val_image):
    os.makedirs(folder_path_val_image)
if not os.path.exists(folder_path_test_image):
    os.makedirs(folder_path_test_image)

forder = 'cm_front'
count_json = 0
count_img = 0

for js in os.listdir(forder):
    if count_json < 1600:
        if "json" in forder + js :
            shutil.move(os.path.join(forder, js), './train/json')
            shutil.move(os.path.join(forder, js).replace(".json",".png"), './train/image')
            count_json = count_json +1
    else: 
        if count_json <1800 :
            if "json" in forder + js :
                shutil.move(os.path.join(forder, js), './val/json')
                shutil.move(os.path.join(forder, js).replace(".json",".png"), './val/image')
                count_json = count_json +1
        else:
            if "json" in forder + js :
                shutil.move(os.path.join(forder, js), './test/json')
                shutil.move(os.path.join(forder, js).replace(".json",".png"), './test/image')
                count_json = count_json +1

#print("CM_Front has 1600 data train, 200 data val," + count - 1800 + " data test")

forder = 'cm_back'
count_json = 0

for js in os.listdir(forder):
    if count_json < 1600:
        if "json" in forder + js :
            shutil.move(os.path.join(forder, js), './train/json')
            shutil.move(os.path.join(forder, js).replace(".json",".png"), './train/image')
            count_json = count_json +1
    else: 
        if count_json <1800 :
            if "json" in forder + js :
                shutil.move(os.path.join(forder, js), './val/json')
                shutil.move(os.path.join(forder, js).replace(".json",".png"), './val/image')
                count_json = count_json +1
        else:
            if "json" in forder + js :
                shutil.move(os.path.join(forder, js), './test/json')
                shutil.move(os.path.join(forder, js).replace(".json",".png"), './test/image')
                count_json = count_json +1

#print("CM_Back has 1600 data train, 200 data val," + count - 1800 + " data test")

forder = 'cc_2_front'
count_json = 0

for js in os.listdir(forder):
    if count_json < 1600:
        if "json" in forder + js :
            shutil.move(os.path.join(forder, js), './train/json')
            shutil.move(os.path.join(forder, js).replace(".json",".png"), './train/image')
            count_json = count_json +1
    else: 
        if count_json <1800 :
            if "json" in forder + js :
                shutil.move(os.path.join(forder, js), './val/json')
                shutil.move(os.path.join(forder, js).replace(".json",".png"), './val/image')
                count_json = count_json +1
        else:
            if "json" in forder + js :
                shutil.move(os.path.join(forder, js), './test/json')
                shutil.move(os.path.join(forder, js).replace(".json",".png"), './test/image')
                count_json = count_json +1
#print("CM_2_Front has 1600 data train, 200 data val," + count - 1800 + " data test")

forder = 'dl_front'
count_json = 0

for js in os.listdir(forder):
    if count_json < 1600:
        if "json" in forder + js :
            shutil.move(os.path.join(forder, js), './train/json')
            shutil.move(os.path.join(forder, js).replace(".json",".png"), './train/image')
            count_json = count_json +1
    else: 
        if count_json <1800 :
            if "json" in forder + js :
                shutil.move(os.path.join(forder, js), './val/json')
                shutil.move(os.path.join(forder, js).replace(".json",".png"), './val/image')
                count_json = count_json +1
        else:
            if "json" in forder + js :
                shutil.move(os.path.join(forder, js), './test/json')
                shutil.move(os.path.join(forder, js).replace(".json",".png"), './test/image')
                count_json = count_json +1
#print("DL_Front has 1600 data train, 200 data val," + count - 1800 + " data test")

forder = 'cc_back'
count_json = 0

for js in os.listdir(forder):
    if count_json < 1600:
        if "json" in forder + js :
            shutil.move(os.path.join(forder, js), './train/json')
            shutil.move(os.path.join(forder, js).replace(".json",".png"), './train/image')
            count_json = count_json +1
    else: 
        if count_json <1800 :
            if "json" in forder + js :
                shutil.move(os.path.join(forder, js), './val/json')
                shutil.move(os.path.join(forder, js).replace(".json",".png"), './val/image')
                count_json = count_json +1
        else:
            if "json" in forder + js :
                shutil.move(os.path.join(forder, js), './test/json')
                shutil.move(os.path.join(forder, js).replace(".json",".png"), './test/image')
                count_json = count_json +1

forder = 'cc_chip_front'
count_json = 0
count_img = 0

for js in os.listdir(forder):
    if count_json < 1600:
        if "json" in forder + js :
            shutil.move(os.path.join(forder, js), './train/json')
            shutil.move(os.path.join(forder, js).replace(".json",".png"), './train/image')
            count_json = count_json +1
    else: 
        if count_json <1800 :
            if "json" in forder + js :
                shutil.move(os.path.join(forder, js), './val/json')
                shutil.move(os.path.join(forder, js).replace(".json",".png"), './val/image')
                count_json = count_json +1
        else:
            if "json" in forder + js :
                shutil.move(os.path.join(forder, js), './test/json')
                shutil.move(os.path.join(forder, js).replace(".json",".png"), './test/image')
                count_json = count_json +1

forder = 'cc_chip_back'
count_json = 0
count_img = 0

for js in os.listdir(forder):
    if count_json < 1600:
        if "json" in forder + js :
            shutil.move(os.path.join(forder, js), './train/json')
            shutil.move(os.path.join(forder, js).replace(".json",".png"), './train/image')
            count_json = count_json +1
    else: 
        if count_json <1800 :
            if "json" in forder + js :
                shutil.move(os.path.join(forder, js), './val/json')
                shutil.move(os.path.join(forder, js).replace(".json",".png"), './val/image')
                count_json = count_json +1
        else:
            if "json" in forder + js :
                shutil.move(os.path.join(forder, js), './test/json')
                shutil.move(os.path.join(forder, js).replace(".json",".png"), './test/image')
                count_json = count_json +1