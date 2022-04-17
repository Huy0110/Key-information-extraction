import pandas as pd
from PIL import Image
import json
import random
from PIL import ImageDraw, ImageFont
from os import listdir

train = pd.read_pickle('train.pkl')
val = pd.read_pickle('dev.pkl')
test = pd.read_pickle('test.pkl')

#Change the k to get another visualize
k = 11005

from collections import Counter

all_labels = [item for sublist in train[1] for item in sublist] + [item for sublist in val[1] for item in sublist] + [item for sublist in test[1] for item in sublist]
print(Counter(all_labels))

replacing_labels = {'address_date':'noi_cap','key_ngay_thang_nam_cap':'key_ngay_cap','key_ngon_tro_phai':'O','value_info_1':'value_info','value_info_2':'value_info','value_info_3':'value_info','nguyen_quan_1':'nguyen_quan','nguyen_quan_2':'nguyen_quan','key_ngay_het_han_en':'key_ngay_het_han','nam_cap':'ngay_cap', 'thang_cap': 'ngay_cap','key_nam_cap':'key_ngay_cap','key_ngay_cap':'key_ngay_cap','key_thang_cap':'key_ngay_cap','ho_ten_1':'ho_ten','ho_ten_2':'ho_ten','ho_khau_thuong_tru_1':'ho_khau_thuong_tru','ho_khau_thuong_tru_2':'ho_khau_thuong_tru','chxh_2_en':'O','key_ngon_tro_trai_en':'O','key_ngon_tro_phai_en':'O','value_nguoi_cap':'O','key_ngon_tro_trai':'O','cnxh_1_en': 'O','chxh_1_en':'O','ignore_1_en': 'O', 'ignore_2_en':'O', 'noi_cap_1':'O', 'noi_cap_2': 'O', 'driver_license_bo_gtvt_1': 'O', 'driver_license_bo_gtvt_2': 'O', 'dau_vet_1': 'dau_vet', 'dau_vet_2': 'dau_vet', 'ignore_1': 'O', 'ignore_2': 'O', 'chxh_1': 'O', 'chxh_2': 'O', 'key_ho_ten_khac':'O', 'level_nguoi_cap_1': 'O', 'level_nguoi_cap_2': 'O', 'nguoi_cap': 'O','van_tay_phai':'O','van_tay_trai':'O'}

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
labels = sorted(labels)

print("LABEL:")
print(labels)

label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for idx, label in enumerate(labels)}

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
#labels = ['O', 'address_date', 'dan_toc', 'dau_vet', 'gioi_tinh', 'ho_khau_thuong_tru_1', 'ho_khau_thuong_tru_2', 'ho_ten', 'ho_ten_1', 'ho_ten_2', 'id', 'key_dan_toc', 'key_dau_vet', 'key_gioi_tinh', 'key_ho_khau_thuong_tru', 'key_ho_ten', 'key_id', 'key_nam_cap', 'key_ngay_cap', 'key_ngay_het_han', 'key_ngay_het_han_en', 'key_ngay_sinh', 'key_ngay_thang_nam_cap', 'key_ngon_tro_phai', 'key_nguyen_quan', 'key_quoc_tich', 'key_thang_cap', 'key_ton_giao', 'nam_cap', 'ngay_cap', 'ngay_het_han', 'ngay_sinh', 'nguyen_quan_1', 'nguyen_quan_2', 'noi_cap', 'quoc_tich', 'thang_cap', 'title', 'title_en', 'ton_giao', 'value_info_1', 'value_info_2', 'value_info_3']

get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
colors = get_colors(len(labels))
draw = ImageDraw.Draw(image, "RGBA")
font = ImageFont.load_default()
label2color = {label: colors[idx] for idx, label in enumerate(labels)}

words, labels, boxes = train

for i in range(len(train[0][k])):
  word = words[k][i]
  label = labels[k][i]
  box = boxes[k][i]

  draw.rectangle(box, outline=label2color[label], width=2)
  #draw.text((box[0]+10, box[1]+5), label, fill=label2color[label], font=font)
  draw.text((box[0]+5, box[1]+10), label, fill=label2color[label], font=font)

image.save('anh1.png')


