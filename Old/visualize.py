import pandas as pd
from PIL import Image
import json
import random
from PIL import ImageDraw, ImageFont
from os import listdir

train = pd.read_pickle('train.pkl')
val = pd.read_pickle('dev.pkl')
test = pd.read_pickle('test.pkl')

k = 0

from collections import Counter

all_labels = [item for sublist in train[1] for item in sublist] + [item for sublist in val[1] for item in sublist] + [item for sublist in test[1] for item in sublist]
print(Counter(all_labels))

replacing_labels = {'ignore_1': 'O', 'ignore_2': 'O', 'chxh_1': 'O', 'chxh_2': 'O', 'key_ho_ten_khac':'O', 'thang_cap': 'O', 'level_nguoi_cap_1': 'O', 'level_nguoi_cap_2': 'O', 'nam_cap': 'O', 'key_nam_cap': 'O', 'key_thang_cap': 'O', 'key_hang_cap': 'O', 'nguoi_cap': 'O', 'hang_cap': 'O','van_tay_phai':'O', 'van_tay_trai': 'O'}

def replace_elem(elem):
  try:
    return replacing_labels[elem]
  except KeyError:
    return elem

def replace_list(ls):
  return [replace_elem(elem) for elem in ls]

train[1] = [replace_list(ls) for ls in train[1]]
val[1] = [replace_list(ls) for ls in val[1]]
test[1] = [replace_list(ls) for ls in test[1]]
all_labels = [item for sublist in train[1] for item in sublist] + [item for sublist in val[1] for item in sublist] + [item for sublist in test[1] for item in sublist]
labels = list(set(all_labels))

print("LABEL:")
print(labels)

label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for idx, label in enumerate(labels)}
#print(label2id)
#print(id2label)

#print(train[0][0])
#print(train[1][0])
#print(train[2][0])
#Vissual

image_dir = 'train/image/'
list_dir_image = listdir(image_dir)
list_dir_image = sorted(list_dir_image) 
image_file_names = [f for f in list_dir_image]

path_w = 'myfile_Train_image.txt'
with open(path_w, mode='a') as f:
  f.writelines('\n'.join(image_file_names))

print("Link anh:")
print(image_dir + image_file_names[k])

image = Image.open(image_dir + image_file_names[k])
image.save('anhChua1.png')
labels = ['key_ngay_sinh', 'ho_khau_thuong_tru_2', 'title', 'key_quoc_tich', 'dau_vet_2', 'quoc_tich', 'key_ho_ten', 'ho_khau_thuong_tru_1', 'noi_cap', 'key_ngay_het_han', 'ton_giao', 'ngay_sinh', 'driver_license_bo_gtvt_2', 'ngay_cap', 'dau_vet_1', 'nguyen_quan_1', 'key_ho_khau_thuong_tru', 'nguyen_quan_2', 'dan_toc', 'O', 'noi_cap_2', 'key_nguyen_quan', 'key_dan_toc', 'key_id', 'noi_cap_1', 'ho_ten_2', 'ho_ten_1', 'ngay_het_han', 'key_ton_giao', 'gioi_tinh', 'address_date', 'key_ngay_cap', 'key_gioi_tinh', 'driver_license_bo_gtvt_1', 'key_dau_vet', 'ho_ten', 'id']

get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
colors = get_colors(len(labels))
#print("COLOR:")
#print(colors)
draw = ImageDraw.Draw(image, "RGBA")

font = ImageFont.load_default()

label2color = {label: colors[idx] for idx, label in enumerate(labels)}

words, labels, boxes = train
#print(train.shape)
for i in range(len(train[0][k])):
  #print("word:")
  word = words[k][i]
  #print(word)
  #print("Label:")
  label = labels[k][i]
  #print(label)
  #print("Box:")
  box = boxes[k][i]
  #print(box)
  draw.rectangle(box, outline=label2color[label], width=2)
  #draw.text((box[0]+10, box[1]+5), label, fill=label2color[label], font=font)
  draw.text((box[0], box[1]), label, fill=label2color[label], font=font)

image.save('anh1.png')


