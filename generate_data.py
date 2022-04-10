from PIL import Image
import json
import random
from PIL import ImageDraw, ImageFont
from matplotlib.font_manager import json_load
from collections import Counter



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

    if("train" in path):
        path_w = 'myfile_Train_anno.txt'
        with open(path_w, mode='w') as f:
            f.writelines('\n'.join(file_name))
    if("val" in path):
        path_w = 'myfile_Val_anno.txt'
        with open(path_w, mode='w') as f:
            f.writelines('\n'.join(file_name))
    if("test" in path):
        path_w = 'myfile_Test_anno.txt'
        with open(path_w, mode='w') as f:
            f.writelines('\n'.join(file_name))


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
        
        # width, height = js['meta']['image_size']['width'], js['meta']['image_size']['height']
        # loop over OCR annotations
        #pre_length = 0
        for elem in js:
            pre_length = 0
            txt = elem['text'].split()
            full_length = len(elem['text'])

            # get bounding box
            # important: each bounding box should be in (upper left, lower right) format
            # it took me some time to understand the upper left is (x1, y3)
            # and the lower right is (x3, y1)
            td1, td2, td3, td4 = elem['polygon']
            xx1, yy1 = td1
            xx2, yy2 = td2
            xx3, yy3 = td3
            xx4, yy4 = td4

            #print("\nSTART A SEGMENT")
            for tu in txt:
                x1 = xx1 + pre_length / full_length * (xx4-xx1)
                y1 = yy1
                x3 = x1 + len(tu) / full_length * (xx4-xx1)
                y3 = yy3
                pre_length = pre_length + len(tu) +1
                box = [x1, y1, x3, y3]
                '''
                if count == 999:
                    print("\nSTART")
                    print(width)
                    print(height)
                    print(td1)
                    print(td2)
                    print(td3)
                    print(td4)
                    print(box)
                    print("END\n")
                '''
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


train_path = 'train/json'
val_path = 'val/json'
test_path = 'test/json'

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




