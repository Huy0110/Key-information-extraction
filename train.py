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
from clearml import Task
from torch.utils.tensorboard import SummaryWriter
from data_loader import Data_Loader
import argparse
from transformers.models.layoutxlm.processing_layoutxlm import LayoutXLMProcessor

# required arg

parser = argparse.ArgumentParser()
   
parser.add_argument('--forder_best', required=True, type=str)
parser.add_argument('--forder_latest', required=True, type=str)
parser.add_argument('--forder_epoch', required=True, type=str)

parser.add_argument('--eval_freq', required=True, type=int)
parser.add_argument('--latest_freq', required=True, type=int)
parser.add_argument('--display_freq', required=True, type=int)
parser.add_argument('--num_train_epochs', required=True, type=int)

forder_best = vars(parser.parse_args())['forder_best']
forder_latest = vars(parser.parse_args())['forder_latest']
forder_epoch = vars(parser.parse_args())['forder_epoch']

eval_freq = vars(parser.parse_args())['eval_freq']
latest_freq = vars(parser.parse_args())['latest_freq']
display_freq = vars(parser.parse_args())['display_freq']
num_train_epochs = vars(parser.parse_args())['num_train_epochs']


if not os.path.exists(forder_best):
    os.makedirs(forder_best)
if not os.path.exists(forder_latest):
    os.makedirs(forder_latest)
if not os.path.exists(forder_best):
    os.makedirs(forder_best)

writer = SummaryWriter("runs")
task = Task.init(project_name='KIE', task_name='model XLM')


def display_val(model, val_dataloader, writer, global_step):
  losses = []
  running_loss_val = 0.0
  for batch in tqdm(val_dataloader, desc="Evaluating"):
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

        loss = outputs.loss
        losses.append(loss.item())
  arg_loss = sum(losses) / len(losses)
  writer.add_scalar('val_loss', arg_loss, global_step)
  return arg_loss

replacing_labels = {'address_date':'noi_cap','key_ngay_thang_nam_cap':'key_ngay_cap','key_ngon_tro_phai':'O','value_info_1':'value_info','value_info_2':'value_info','value_info_3':'value_info','nguyen_quan_1':'nguyen_quan','nguyen_quan_2':'nguyen_quan','key_ngay_het_han_en':'key_ngay_het_han','nam_cap':'ngay_cap', 'thang_cap': 'ngay_cap','key_nam_cap':'key_ngay_cap','key_ngay_cap':'key_ngay_cap','key_thang_cap':'key_ngay_cap','ho_ten_1':'ho_ten','ho_ten_2':'ho_ten','ho_khau_thuong_tru_1':'ho_khau_thuong_tru','ho_khau_thuong_tru_2':'ho_khau_thuong_tru','chxh_2_en':'O','key_ngon_tro_trai_en':'O','key_ngon_tro_phai_en':'O','value_nguoi_cap':'O','key_ngon_tro_trai':'O','cnxh_1_en': 'O','chxh_1_en':'O','ignore_1_en': 'O', 'ignore_2_en':'O', 'noi_cap_1':'O', 'noi_cap_2': 'O', 'driver_license_bo_gtvt_1': 'O', 'driver_license_bo_gtvt_2': 'O', 'dau_vet_1': 'dau_vet', 'dau_vet_2': 'dau_vet', 'ignore_1': 'O', 'ignore_2': 'O', 'chxh_1': 'O', 'chxh_2': 'O', 'key_ho_ten_khac':'O', 'level_nguoi_cap_1': 'O', 'level_nguoi_cap_2': 'O', 'nguoi_cap': 'O','van_tay_phai':'O','van_tay_trai':'O'}

#processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutxlm-base", revision="no_ocr")
processor = LayoutXLMProcessor.from_pretrained("microsoft/layoutxlm-base")
processor.feature_extractor.apply_ocr = False
data_loader = Data_Loader(replacing_labels = replacing_labels,processor = processor)
train_dataloader, val_dataloader, test_dataloader, label2id, id2label, labels, all_labels = data_loader.load_data()


#model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutxlm-base',num_labels=len(labels))
model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutxlm-base', num_labels=len(labels))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

global_step = 0
loss_best = 100


#put the model in training mode
running_loss = 0.0
n_total_steps = len(train_dataloader)
model.train() 
for epoch in range(num_train_epochs):  
   print("Epoch:", epoch)
   for batch in tqdm(train_dataloader):
        # get the inputs;
        input_ids = batch['input_ids'].to(device)
        bbox = batch['bbox'].to(device)
        image = batch['image'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = model(input_ids=input_ids,
                        bbox=bbox,
                        image=image,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels) 
        loss = outputs.loss
        running_loss += loss.item()
        
        # print loss every 100 steps
        if global_step % display_freq == 0:
          print(f"Loss after {global_step} steps: {loss.item()}")
          writer.add_scalar('training_loss', running_loss / 100, global_step)
          running_loss = 0.0

        if global_step % eval_freq == 0 and global_step >= eval_freq:
          model.eval()
          loss_eval = display_val(model, val_dataloader, writer, global_step)
          print("Display val:")
          print(loss_eval)
          if(loss_eval < loss_best):
            loss_best = loss_eval
            model.save_pretrained(forder_best)
            print("Saving the best model")
          model.train()
        
        if global_step % latest_freq == 0 and global_step >= latest_freq:
          model.save_pretrained(forder_latest)
          print("Saving the latest model")

        loss.backward()
        optimizer.step()
        global_step += 1
   model.save_pretrained(forder_epoch)
   print("Saving the latest epoch model")

#model.save_pretrained("Checkpoints")