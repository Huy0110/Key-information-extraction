from PIL import Image
import json
import random
from PIL import ImageDraw, ImageFont
from matplotlib.font_manager import json_load
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path_image', required=True, type=str)
parser.add_argument('--path_json', required=True, type=str)
parser.add_argument('--path_save', required=True, type=str)

path_image = vars(parser.parse_args())['path_image']
path_json = vars(parser.parse_args())['path_json']
path_save = vars(parser.parse_args())['path_save']


from tqdm.notebook import tqdm
import os


def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]


def generate_annotations(path_image: str, path_json: str):
    annotation_files = []
    file_name = []
    count = 0
    with open(path_json) as f:
        js = json.load(f)

    words = []
    boxes = []
    labels = []
    words_example = []
    boxes_example = []
    labels_example = []
    im = Image.open(path_image)
    count = count + 1
    width, height = im.size
    for elem in js:
        pre_length = 0
        txt = elem['text'].split()
        full_length = len(elem['text'])

        td1, td4, td3, td2 = elem['polygon']
        xx1, yy1 = td1
        xx2, yy2 = td2
        xx3, yy3 = td3
        xx4, yy4 = td4

        for tu in txt:
            x1 = xx1 + pre_length / full_length * (xx4-xx1)
            y1 = yy1
            x3 = x1 + len(tu) / full_length * (xx4-xx1)
            y3 = yy3
            pre_length = pre_length + len(tu) +1
            box = [x1, y1, x3, y3]

            box = normalize_bbox(box, width=width, height=height)
            if len(tu) < 1:
                    continue
            if min(box) < 0 or max(box) >1000:  # another bug in which a box had -4
                    continue
                # another bug in which a box difference was -12
            if ((box[3] - box[1]) < 0) or ((box[2] - box[0]) < 0):
                    continue
                # ADDED
            words_example.append(tu)
            labels_example.append(elem['key'])
            boxes_example.append(box)
    words.append(words_example)
    boxes.append(boxes_example)
    labels.append(labels_example)

    return words, boxes, labels

words, boxes, labels = generate_annotations(path_image,path_json)

import pickle
with open('anno.pkl', 'wb') as t:
    pickle.dump([words, labels, boxes], t)

os.makedirs(path_save, exist_ok= True)
os.makedirs(path_save + '/image', exist_ok = True)
import shutil
shutil.copy('anno.pkl', path_save)
shutil.copy(path_image, path_save + '/image')




