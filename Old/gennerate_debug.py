from PIL import Image
import json
import random
from PIL import ImageDraw, ImageFont
from matplotlib.font_manager import json_load
from collections import Counter
import shutil, os
from tqdm.notebook import tqdm

if not os.path.exists('debug_image_real_3'):
    os.makedirs('debug_image_real_3')


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
    list_dir_debug = os.listdir(path)
    list_dir_debug = sorted(list_dir_debug)
    count = 0
    for list_dir in tqdm(list_dir_debug):
        #print(list_dir)
        objects = os.listdir(os.path.join("debugs",list_dir))
        files_file = [f for f in objects if os.path.isfile(os.path.join("debugs" + "/" +list_dir, f))]
        #print(files_file)
        for js in tqdm(files_file):
            #print(list_dir + "/"+js)
            if "ocr_file.json" in list_dir + js:
                print("Count")
                with open(os.path.join(path+"/"+list_dir,"ocr_file.json")) as f:
                    annotation_files.append(json.load(f))
                    file_name.append(path+"/"+list_dir)
                    count = count +1
        #print(path+js)

    file_name = sorted(file_name)
    #print(count)


    words = []
    boxes = []
    labels = []
    count =0
    #print(annotation_files)
    for js in tqdm(annotation_files):
        #print("Here")
        words_example = []
        boxes_example = []
        labels_example = []
        fn = file_name[count]
        #print(fn)
        fn = os.path.join(fn, "debug.png")
        print(fn)
        shutil.copy(fn, 'debug_image_real_3')
        fn1 = fn.replace("/", "_")
        os.rename('debug_image_real_3/debug.png','debug_image_real_3/'+fn1)

        #print(fn)
        im = Image.open(fn)
        count = count + 1
        width, height = im.size
        
        # width, height = js['meta']['image_size']['width'], js['meta']['image_size']['height']
        # loop over OCR annotations
        #pre_length = 0
        for elem in js["textlines"]:
            pre_length = 0
            txt = elem['text'].split()
            full_length = len(elem['text'])

            # get bounding box
            # important: each bounding box should be in (upper left, lower right) format
            # it took me some time to understand the upper left is (x1, y3)
            # and the lower right is (x3, y1)
            td1, td2, td3, td4 = elem['polys']
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
                #box = normalize_bbox(box, width=width, height=height)
                if len(tu) < 1:
                        continue
                #if min(box) < 0 or max(box) >1000:  # another bug in which a box had -4
                #        continue
                    # another bug in which a box difference was -12
                if ((box[3] - box[1]) < 0) or ((box[2] - box[0]) < 0):
                        continue
                    # ADDED
                words_example.append(tu)
                labels_example.append("O")
                boxes_example.append(box)
            #print("END A SEGMENT\n")
        words.append(words_example)
        boxes.append(boxes_example)
        labels.append(labels_example)

    return words, boxes, labels

path_debug = 'debugs'

words_debug, boxes_debug, labels_debug = generate_annotations(path_debug)

all_labels = [item for sublist in labels_debug for item in sublist]
print(Counter(all_labels))



import pickle
with open('debug.pkl', 'wb') as t:
    pickle.dump([words_debug, labels_debug, boxes_debug], t)




