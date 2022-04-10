from PIL import Image
import json
import random
from PIL import ImageDraw, ImageFont
from matplotlib.font_manager import json_load
from collections import Counter
import shutil, os
from tqdm.notebook import tqdm
import pickle

if not os.path.exists('debug_pkl_real'):
    os.makedirs('debug_pkl_real')


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
        files_file = sorted(files_file)
        for js in tqdm(files_file):
            #print(list_dir + "/"+js)
            if "ocr_file.json" in list_dir + js:
                print("Count")
                words_example = []
                boxes_example = []
                labels_example = []
                words = []
                boxes = []
                labels = []
                with open(os.path.join(path+"/"+list_dir,"ocr_file.json")) as f:
                    f_json = json.load(f)
                    f_name = path + "/" + list_dir
                    count = count +1
                    fn = os.path.join(f_name, "debug.png")
                    print(fn)
                    im = Image.open(fn)
                    width, height = im.size
                    for elem in f_json["textlines"]:
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
                            labels_example.append("O")
                            boxes_example.append(box)
                    #print("END A SEGMENT\n")
                    words.append(words_example)
                    boxes.append(boxes_example)
                    labels.append(labels_example)
                    path_save = 'debug_pkl_real/debug' + '_' + str(count)
                    os.makedirs(path_save, exist_ok = True)
                    os.makedirs(os.path.join(path_save,'image'), exist_ok = True)
                    with open(path_save + '/anno.'+'pkl', 'wb') as t:
                        pickle.dump([words, labels, boxes], t)
                    shutil.copy(fn, os.path.join(path_save,'image'))


path_debug = 'debugs'

generate_annotations(path_debug)
