import pandas as pd

train = pd.read_pickle('train.pkl')
val = pd.read_pickle('dev.pkl')
test = pd.read_pickle('test.pkl')

from collections import Counter

all_labels = [item for sublist in train[1] for item in sublist] + [item for sublist in val[1] for item in sublist] + [item for sublist in test[1] for item in sublist]
#print(Counter(all_labels))

replacing_labels = {'chxh_1': 'O', 'chxh_2': 'O', 'key_ho_ten_khac':'O', 'thang_cap': 'O', 'level_nguoi_cap_1': 'O', 'level_nguoi_cap_2': 'O', 'nam_cap': 'O', 'key_nam_cap': 'O', 'key_thang_cap': 'O', 'key_hang_cap': 'O', 'nguoi_cap': 'O', 'hang_cap': 'O','van_tay_phai':'O', 'van_tay_trai': 'O'}

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

print(labels)

label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for idx, label in enumerate(labels)}
#print(label2id)
#print(id2label)

#print(train[0][0])
#print(train[1][0])
#Vissual
