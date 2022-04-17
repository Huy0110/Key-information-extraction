import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import LayoutLMv2ForTokenClassification, AdamW
from transformers import LayoutLMv2Processor
import torch
from tqdm.notebook import tqdm
from os import listdir
from PIL import Image
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score)
from DatasetIDCard import DatasetIDCard
from transformers.models.layoutxlm.processing_layoutxlm import LayoutXLMProcessor

def replace_elem(elem):
  try:
    return replacing_labels[elem]
  except KeyError:
    return elem

def replace_list(ls):
  return [replace_elem(elem) for elem in ls]

replacing_labels = {'address_date':'noi_cap','key_ngay_thang_nam_cap':'key_ngay_cap','key_ngon_tro_phai':'O','value_info_1':'value_info','value_info_2':'value_info','value_info_3':'value_info','nguyen_quan_1':'nguyen_quan','nguyen_quan_2':'nguyen_quan','key_ngay_het_han_en':'key_ngay_het_han','nam_cap':'ngay_cap', 'thang_cap': 'ngay_cap','key_nam_cap':'key_ngay_cap','key_ngay_cap':'key_ngay_cap','key_thang_cap':'key_ngay_cap','ho_ten_1':'ho_ten','ho_ten_2':'ho_ten','ho_khau_thuong_tru_1':'ho_khau_thuong_tru','ho_khau_thuong_tru_2':'ho_khau_thuong_tru','chxh_2_en':'O','key_ngon_tro_trai_en':'O','key_ngon_tro_phai_en':'O','value_nguoi_cap':'O','key_ngon_tro_trai':'O','cnxh_1_en': 'O','chxh_1_en':'O','ignore_1_en': 'O', 'ignore_2_en':'O', 'noi_cap_1':'O', 'noi_cap_2': 'O', 'driver_license_bo_gtvt_1': 'O', 'driver_license_bo_gtvt_2': 'O', 'dau_vet_1': 'dau_vet', 'dau_vet_2': 'dau_vet', 'ignore_1': 'O', 'ignore_2': 'O', 'chxh_1': 'O', 'chxh_2': 'O', 'key_ho_ten_khac':'O', 'level_nguoi_cap_1': 'O', 'level_nguoi_cap_2': 'O', 'nguoi_cap': 'O','van_tay_phai':'O','van_tay_trai':'O'}
#processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base")
processor.feature_extractor.apply_ocr = False
class Data_Loader():
    def __init__(self,
                 path_train = 'train.pkl', 
                 path_val = 'dev.pkl', 
                 path_test = 'test.pkl',
                 replacing_labels = replacing_labels,
                 processor = processor,
                 image_dir_train = 'train/image/',
                 image_dir_val = 'val/image/',
                 image_dir_test = 'test/image/'):
        self.path_train = path_train
        self.path_val = path_val
        self.path_test = path_test
        self.replacing_labels = replacing_labels
        self.processor = processor
        self.image_dir_train = image_dir_train
        self.image_dir_val = image_dir_val 
        self.image_dir_test = image_dir_test 
        self.replace_list = replace_list


    def load_data(self):
        train = pd.read_pickle(self.path_train)
        val = pd.read_pickle(self.path_val)
        test = pd.read_pickle(self.path_test)
        all_labels = [item for sublist in train[1] for item in sublist] + [item for sublist in val[1] for item in sublist] + [item for sublist in test[1] for item in sublist]
        train[1] = [self.replace_list(ls) for ls in train[1]]
        val[1] = [self.replace_list(ls) for ls in val[1]]
        test[1] = [self.replace_list(ls) for ls in test[1]]
        all_labels = [item for sublist in train[1] for item in sublist] + [item for sublist in val[1] for item in sublist] + [item for sublist in test[1] for item in sublist]
        labels = list(set(all_labels))
        labels = sorted(labels)
        print(labels)
        label2id = {label: idx for idx, label in enumerate(labels)}
        id2label = {idx: label for idx, label in enumerate(labels)}
        print(label2id)
        train_dataset = DatasetIDCard(annotations=train,
                            image_dir=self.image_dir_train, 
                            processor=self.processor,
                            label2id = label2id)
        val_dataset = DatasetIDCard(annotations=val,
                            image_dir=self.image_dir_val, 
                            processor=self.processor,
                            label2id = label2id)
        test_dataset = DatasetIDCard(annotations=test,
                            image_dir=self.image_dir_test, 
                            processor=self.processor,
                            label2id = label2id)
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=4)
        return train_dataloader, val_dataloader, test_dataloader, label2id, id2label, labels, all_labels