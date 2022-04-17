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
from data_loader import Data_Loader
import argparse
from transformers.models.layoutxlm.processing_layoutxlm import LayoutXLMProcessor

# required arg

parser = argparse.ArgumentParser()
   
parser.add_argument('--path_check', required=True, type=str)

path_check = vars(parser.parse_args())['path_check']

replacing_labels = {'address_date':'noi_cap','key_ngay_thang_nam_cap':'key_ngay_cap','key_ngon_tro_phai':'O','value_info_1':'value_info','value_info_2':'value_info','value_info_3':'value_info','nguyen_quan_1':'nguyen_quan','nguyen_quan_2':'nguyen_quan','key_ngay_het_han_en':'key_ngay_het_han','nam_cap':'ngay_cap', 'thang_cap': 'ngay_cap','key_nam_cap':'key_ngay_cap','key_ngay_cap':'key_ngay_cap','key_thang_cap':'key_ngay_cap','ho_ten_1':'ho_ten','ho_ten_2':'ho_ten','ho_khau_thuong_tru_1':'ho_khau_thuong_tru','ho_khau_thuong_tru_2':'ho_khau_thuong_tru','chxh_2_en':'O','key_ngon_tro_trai_en':'O','key_ngon_tro_phai_en':'O','value_nguoi_cap':'O','key_ngon_tro_trai':'O','cnxh_1_en': 'O','chxh_1_en':'O','ignore_1_en': 'O', 'ignore_2_en':'O', 'noi_cap_1':'O', 'noi_cap_2': 'O', 'driver_license_bo_gtvt_1': 'O', 'driver_license_bo_gtvt_2': 'O', 'dau_vet_1': 'dau_vet', 'dau_vet_2': 'dau_vet', 'ignore_1': 'O', 'ignore_2': 'O', 'chxh_1': 'O', 'chxh_2': 'O', 'key_ho_ten_khac':'O', 'level_nguoi_cap_1': 'O', 'level_nguoi_cap_2': 'O', 'nguoi_cap': 'O','van_tay_phai':'O','van_tay_trai':'O'}
#processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base")
processor.feature_extractor.apply_ocr = False

data_loader = Data_Loader(replacing_labels = replacing_labels,processor = processor)
train_dataloader, val_dataloader, test_dataloader, label2id, id2label, labels, all_labels = data_loader.load_data()

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
for batch in tqdm(test_dataloader, desc="Evaluating"):
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

print("END VAL")
print("START TEST")

import warnings
warnings.filterwarnings("ignore")

def results_test(preds, out_label_ids, labels):
  preds = np.argmax(preds, axis=2)

  label_map = {i: label for i, label in enumerate(labels)}

  out_label_list = [[] for _ in range(out_label_ids.shape[0])]
  preds_list = [[] for _ in range(out_label_ids.shape[0])]

  for i in range(out_label_ids.shape[0]):
      for j in range(out_label_ids.shape[1]):
          if out_label_ids[i, j] != -100:
              out_label_list[i].append(label_map[out_label_ids[i][j]])
              preds_list[i].append(label_map[preds[i][j]])

  results = {
      "precision": precision_score(out_label_list, preds_list),
      "recall": recall_score(out_label_list, preds_list),
      "f1": f1_score(out_label_list, preds_list),
  }
  return results, classification_report(out_label_list, preds_list)

labels = list(set(all_labels))
val_result, class_report = results_test(preds_val, out_label_ids, labels)
print("Overall results:", val_result)
print(class_report)