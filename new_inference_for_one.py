import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import LayoutLMv2ForTokenClassification, AdamW
from transformers import LayoutLMv2Processor
import torch
from tqdm.notebook import tqdm
from os import listdir
import os
from PIL import Image
import json
from data_loader import Data_Loader
from DatasetIDCard import DatasetIDCard
import argparse
from transformers.models.layoutxlm.processing_layoutxlm import LayoutXLMProcessor

# required arg

parser = argparse.ArgumentParser()
parser.add_argument('--path_test', required=True, type=str)
parser.add_argument('--path_check', required=True, type=str)

path_test = vars(parser.parse_args())['path_test']
path_check = vars(parser.parse_args())['path_check']
print(path_test)

replacing_labels = {'address_date':'noi_cap','key_ngay_thang_nam_cap':'key_ngay_cap','key_ngon_tro_phai':'O','value_info_1':'value_info','value_info_2':'value_info','value_info_3':'value_info','nguyen_quan_1':'nguyen_quan','nguyen_quan_2':'nguyen_quan','key_ngay_het_han_en':'key_ngay_het_han','nam_cap':'ngay_cap', 'thang_cap': 'ngay_cap','key_nam_cap':'key_ngay_cap','key_ngay_cap':'key_ngay_cap','key_thang_cap':'key_ngay_cap','ho_ten_1':'ho_ten','ho_ten_2':'ho_ten','ho_khau_thuong_tru_1':'ho_khau_thuong_tru','ho_khau_thuong_tru_2':'ho_khau_thuong_tru','chxh_2_en':'O','key_ngon_tro_trai_en':'O','key_ngon_tro_phai_en':'O','value_nguoi_cap':'O','key_ngon_tro_trai':'O','cnxh_1_en': 'O','chxh_1_en':'O','ignore_1_en': 'O', 'ignore_2_en':'O', 'noi_cap_1':'O', 'noi_cap_2': 'O', 'driver_license_bo_gtvt_1': 'O', 'driver_license_bo_gtvt_2': 'O', 'dau_vet_1': 'dau_vet', 'dau_vet_2': 'dau_vet', 'ignore_1': 'O', 'ignore_2': 'O', 'chxh_1': 'O', 'chxh_2': 'O', 'key_ho_ten_khac':'O', 'level_nguoi_cap_1': 'O', 'level_nguoi_cap_2': 'O', 'nguoi_cap': 'O','van_tay_phai':'O','van_tay_trai':'O'}


def replace_elem(elem):
  try:
    return replacing_labels[elem]
  except KeyError:
    return elem
def replace_list(ls):
  return [replace_elem(elem) for elem in ls]

def listToString(s):   
    # initialize an empty string
    #str1 = ""    
    # traverse in the string  
    count = 0
    for ele in s: 
      if count == 0:
        str1 = ele
        count+=1
      else:
        #if "##" in ele:
        #    ele = ele[2:]
        #    str1 +=ele
        #else:
        str1 = str1 + ' ' + ele    
    # return string  
    return str1

#processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base")
processor.feature_extractor.apply_ocr = False
#fake_path = os.path.join(path_test,'anno.pkl')
#fake_path_img = os.path.join(path_test,'image')
#data_loader = Data_Loader(path_train = fake_path, path_val = fake_path,path_test = fake_path,replacing_labels = replacing_labels,processor = processor, image_dir_train = fake_path_img,image_dir_val = fake_path_img,image_dir_test = fake_path_img)
#train_dataloader_fake, val_dataloader_fake, test_dataloader_fake, label2id_fake, id2label_fake, labels_fake, all_labels_fake = data_loader.load_data()

label2id = {'O': 0, 'dan_toc': 1, 'dau_vet': 2, 'gioi_tinh': 3, 'hang_cap': 4, 'ho_khau_thuong_tru': 5, 'ho_ten': 6, 'id': 7, 'key_dan_toc': 8, 'key_dau_vet': 9, 'key_gioi_tinh': 10, 'key_hang_cap': 11, 'key_ho_khau_thuong_tru': 12, 'key_ho_ten': 13, 'key_id': 14, 'key_ngay_cap': 15, 'key_ngay_het_han': 16, 'key_ngay_sinh': 17, 'key_nguyen_quan': 18, 'key_quoc_tich': 19, 'key_ton_giao': 20, 'ngay_cap': 21, 'ngay_het_han': 22, 'ngay_sinh': 23, 'nguyen_quan': 24, 'noi_cap': 25, 'quoc_tich': 26, 'title': 27, 'title_en': 28, 'ton_giao': 29, 'value_info': 30}
id2label = {0: 'O', 1: 'dan_toc', 2: 'dau_vet', 3: 'gioi_tinh', 4: 'hang_cap', 5: 'ho_khau_thuong_tru', 6: 'ho_ten', 7: 'id', 8: 'key_dan_toc', 9: 'key_dau_vet', 10: 'key_gioi_tinh', 11: 'key_hang_cap', 12: 'key_ho_khau_thuong_tru', 13: 'key_ho_ten', 14: 'key_id', 15: 'key_ngay_cap', 16: 'key_ngay_het_han', 17: 'key_ngay_sinh', 18: 'key_nguyen_quan', 19: 'key_quoc_tich', 20: 'key_ton_giao', 21: 'ngay_cap', 22: 'ngay_het_han', 23: 'ngay_sinh', 24: 'nguyen_quan', 25: 'noi_cap', 26: 'quoc_tich', 27: 'title', 28: 'title_en', 29: 'ton_giao', 30: 'value_info'}
labels = ['O', 'dan_toc', 'dau_vet', 'gioi_tinh', 'hang_cap', 'ho_khau_thuong_tru', 'ho_ten', 'id', 'key_dan_toc', 'key_dau_vet', 'key_gioi_tinh', 'key_hang_cap', 'key_ho_khau_thuong_tru', 'key_ho_ten', 'key_id', 'key_ngay_cap', 'key_ngay_het_han', 'key_ngay_sinh', 'key_nguyen_quan', 'key_quoc_tich', 'key_ton_giao', 'ngay_cap', 'ngay_het_han', 'ngay_sinh', 'nguyen_quan', 'noi_cap', 'quoc_tich', 'title', 'title_en', 'ton_giao', 'value_info']

debug = pd.read_pickle(os.path.join(path_test,'anno.pkl'))
debug[1] = [replace_list(ls) for ls in debug[1]]
debug_dataset = DatasetIDCard(annotations=debug,image_dir=os.path.join(path_test, 'image/'),processor=processor,label2id = label2id)
debug_dataloader = DataLoader(debug_dataset, batch_size=1)
numberOfLabel = len(labels)

model = LayoutLMv2ForTokenClassification.from_pretrained(path_check,num_labels=len(labels))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

model.eval()
import numpy as np

preds_val = None
out_label_ids = None

# put model in evaluation mode
model.eval()
Final_pred = {}
Final_result = {}
count =0
for batch in tqdm(debug_dataloader, desc="Evaluating"):
    count = count +1
    with torch.no_grad():
        input_ids = batch['input_ids'].to(device)
        bbox = batch['bbox'].to(device)
        image = batch['image'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        # forward pass
        outputs = model(input_ids=input_ids, bbox=bbox, image=image, attention_mask=attention_mask, 
                        token_type_ids=token_type_ids, labels=labels)
        
        if preds_val is None:
          preds_val = outputs.logits.detach().cpu().numpy()
          out_label_ids = batch["labels"].detach().cpu().numpy()
        else:
          preds_val = np.append(preds_val, outputs.logits.detach().cpu().numpy(), axis=0)
          out_label_ids = np.append(
              out_label_ids, batch["labels"].detach().cpu().numpy(), axis=0
          )
        pred = outputs.logits.argmax(-1).squeeze().tolist()

        for id, label in zip(input_ids.squeeze(), pred):
          if label == 0 or processor.tokenizer.decode(id) == '<pad>'or processor.tokenizer.decode(id) == '<cls>'or processor.tokenizer.decode(id) == '<sep>':
            continue
          if id2label[label] in Final_pred:
            Final_pred[id2label[label]].append(processor.tokenizer.decode(id))
          else:
            Final_pred[id2label[label]] = [processor.tokenizer.decode(id)]

for key in Final_pred:
    Final_result[key] = listToString(Final_pred[key])
#print(Final_result)
with open(os.path.join(path_test, "final.json"), "w",encoding='utf-8') as outfile:
    json.dump(Final_result, outfile,indent = 4,ensure_ascii=False)
#debug_labels = [id2label[idx] for idx in preds_val]