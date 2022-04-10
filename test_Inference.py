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
import json
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score)
from data_loader import Data_Loader

replacing_labels = {'chxh_2_en':'O','key_ngon_tro_trai_en':'O','key_ngon_tro_phai_en':'O','value_nguoi_cap':'O','key_ngon_tro_trai':'O','cnxh_1_en': 'O','chxh_1_en':'O','ignore_1_en': 'O', 'ignore_2_en':'O', 'noi_cap_1':'O', 'noi_cap_2': 'O', 'driver_license_bo_gtvt_1': 'O', 'driver_license_bo_gtvt_2': 'O', 'dau_vet_1': 'dau_vet', 'dau_vet_2': 'dau_vet', 'ignore_1': 'O', 'ignore_2': 'O', 'chxh_1': 'O', 'chxh_2': 'O', 'key_ho_ten_khac':'O', 'level_nguoi_cap_1': 'O', 'level_nguoi_cap_2': 'O', 'key_hang_cap': 'O', 'nguoi_cap': 'O', 'hang_cap': 'O','van_tay_phai':'O','van_tay_trai':'O'}


processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

debug = pd.read_pickle('debug.pkl')
debug_dataset = Dataset(annotations=debug,
                            image_dir='debug_image_real_3/', 
                            processor=processor)
debug_dataloader = DataLoader(debug_dataset, batch_size=1)
data_loader = Data_Loader(replacing_labels = replacing_labels,processor = processor)
train_dataloader, val_dataloader, test_dataloader, label2id, id2label, labels, all_labels = data_loader.load_data()

model = LayoutLMv2ForTokenClassification.from_pretrained('Best',num_labels=len(labels))

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
          if label == 0:
            continue
          if id2label[label] in Final_pred:
            Final_pred[id2label[label]].append(processor.tokenizer.decode(id))
          else:
            Final_pred[id2label[label]] = [processor.tokenizer.decode(id)]
    if count == 1:
      break
print(Final_pred)
with open("sample.json", "w") as outfile:
    json.dump(Final_pred, outfile,indent = 4)
#debug_labels = [id2label[idx] for idx in preds_val]