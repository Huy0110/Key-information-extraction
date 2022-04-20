from PIL import Image
import json
import random
from PIL import ImageDraw, ImageFont
from matplotlib.font_manager import json_load
from collections import Counter
import argparse

# required arg

parser = argparse.ArgumentParser()
   
parser.add_argument('--train_path', required=True, type=str)
parser.add_argument('--val_path', required=True, type=str)
parser.add_argument('--test_path', required=True, type=str)

train_path = vars(parser.parse_args())['train_path']
val_path = vars(parser.parse_args())['val_path']
test_path = vars(parser.parse_args())['test_path']



from tqdm.notebook import tqdm
import os


def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]


def generate_annotations(path: str):
    annotation_files = []
    file_name = []
    list_dir = os.listdir(path)
    list_dir = sorted(list_dir)
    count = 0
    for js in tqdm(list_dir):
        #print(path+js)
        if ".json" in path + js: 
            with open(os.path.join(path, js)) as f:
                annotation_files.append(json.load(f))
                file_name.append(path + js)
                count = count + 1

    file_name = sorted(file_name)
    #print(count)

    words = []
    boxes = []
    labels = []
    count =0
    for js in tqdm(annotation_files):
        words_example = []
        boxes_example = []
        labels_example = []
        fn = file_name[count]
        fn = fn.replace(".json",".png")
        fn = fn.replace("json","image/")
        #print(fn)
        im = Image.open(fn)
        count = count + 1
        width, height = im.size
        
        for elem in js:
            pre_length = 0
            tu = elem['text']
            td1, td2, td3, td4 = elem['polygon']
            xx1, yy1 = td1
            xx2, yy2 = td2
            xx3, yy3 = td3
            xx4, yy4 = td4
            box = [xx1, yy1, xx3, yy3]
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
            #print("END A SEGMENT\n")
        words.append(words_example)
        boxes.append(boxes_example)
        labels.append(labels_example)

    return words, boxes, labels


words_train, boxes_train, labels_train = generate_annotations(train_path)
words_val, boxes_val, labels_val = generate_annotations(val_path)
words_test, boxes_test, labels_test = generate_annotations(test_path)

all_labels = [item for sublist in labels_train for item in sublist]
print(Counter(all_labels))



import pickle
with open('train.pkl', 'wb') as t:
    pickle.dump([words_train, labels_train, boxes_train], t)
with open('dev.pkl', 'wb') as t:
    pickle.dump([words_val, labels_val, boxes_val], t)
with open('test.pkl', 'wb') as t:
    pickle.dump([words_test, labels_test, boxes_test], t)




