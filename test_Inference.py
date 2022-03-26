import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import LayoutLMv2ForTokenClassification, AdamW
import torch
from tqdm.notebook import tqdm
from os import listdir
from PIL import Image
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score)



def replace_elem(elem):
  try:
    return replacing_labels[elem]
  except KeyError:
    return elem

def replace_list(ls):
  return [replace_elem(elem) for elem in ls]

train = pd.read_pickle('train.pkl')
val = pd.read_pickle('dev.pkl')
test = pd.read_pickle('test.pkl')
debug = pd.read_pickle('debug.pkl')
all_labels = [item for sublist in train[1] for item in sublist] + [item for sublist in val[1] for item in sublist] + [item for sublist in test[1] for item in sublist]
#print(Counter(all_labels))
replacing_labels = {'chxh_2_en':'O','key_ngon_tro_trai_en':'O','key_ngon_tro_phai_en':'O','value_nguoi_cap':'O','key_ngon_tro_trai':'O','cnxh_1_en': 'O','chxh_1_en':'O','ignore_1_en': 'O', 'ignore_2_en':'O', 'noi_cap_1':'O', 'noi_cap_2': 'O', 'driver_license_bo_gtvt_1': 'O', 'driver_license_bo_gtvt_2': 'O', 'dau_vet_1': 'dau_vet', 'dau_vet_2': 'dau_vet', 'ignore_1': 'O', 'ignore_2': 'O', 'chxh_1': 'O', 'chxh_2': 'O', 'key_ho_ten_khac':'O', 'level_nguoi_cap_1': 'O', 'level_nguoi_cap_2': 'O', 'key_hang_cap': 'O', 'nguoi_cap': 'O', 'hang_cap': 'O','van_tay_phai':'O','van_tay_trai':'O'}

train[1] = [replace_list(ls) for ls in train[1]]
val[1] = [replace_list(ls) for ls in val[1]]
test[1] = [replace_list(ls) for ls in test[1]]
all_labels = [item for sublist in train[1] for item in sublist] + [item for sublist in val[1] for item in sublist] + [item for sublist in test[1] for item in sublist]
labels = list(set(all_labels))
labels = sorted(labels)
print(labels)

label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for idx, label in enumerate(labels)}
print(label2id)
#print(all_labels)
#print(id2label)

class Dataset(Dataset):
    """CORD dataset."""

    def __init__(self, annotations, image_dir, processor=None, max_length=512):
        """
        Args:
            annotations (List[List]): List of lists containing the word-level annotations (words, labels, boxes).
            image_dir (string): Directory with all the document images.
            processor (LayoutLMv2Processor): Processor to prepare the text + image.
        """
        self.words, self.labels, self.boxes = annotations
        self.image_dir = image_dir
        list_dir_image = listdir(image_dir)
        list_dir_image = sorted(list_dir_image) 
        #image_file_names = [f for f in list_dir_image]
        self.image_file_names = [f for f in list_dir_image]
        self.processor = processor

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        # first, take an image
        item = self.image_file_names[idx]
        image = Image.open(self.image_dir + item).convert("RGB")

        # get word-level annotations 
        words = self.words[idx]
        boxes = self.boxes[idx]
        word_labels = self.labels[idx]

        assert len(words) == len(boxes) == len(word_labels)
        
        word_labels = [label2id[label] for label in word_labels]

        #word_labels = [self.label2id[label] for label in word_labels]
        # use processor to prepare everything
        encoded_inputs = self.processor(image, words, boxes=boxes, word_labels=word_labels, 
                                        padding="max_length", truncation=True, 
                                        return_tensors="pt")
        
        # remove batch dimension
        for k,v in encoded_inputs.items():
          encoded_inputs[k] = v.squeeze()

        assert encoded_inputs.input_ids.shape == torch.Size([512])
        assert encoded_inputs.attention_mask.shape == torch.Size([512])
        assert encoded_inputs.token_type_ids.shape == torch.Size([512])
        assert encoded_inputs.bbox.shape == torch.Size([512, 4])
        assert encoded_inputs.image.shape == torch.Size([3, 224, 224])
        assert encoded_inputs.labels.shape == torch.Size([512]) 
      
        return encoded_inputs

from torch.utils.data import DataLoader
from transformers import LayoutLMv2Processor

processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")

train_dataset = Dataset(annotations=train,
                            image_dir='train/image/', 
                            processor=processor)
val_dataset = Dataset(annotations=val,
                            image_dir='val/image/', 
                            processor=processor)
test_dataset = Dataset(annotations=test,
                            image_dir='test/image/', 
                            processor=processor)
debug_dataset = Dataset(annotations=debug,
                            image_dir='debug_image/', 
                            processor=processor)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2)
debug_dataloader = DataLoader(debug_dataset, batch_size=2)

model = LayoutLMv2ForTokenClassification.from_pretrained('Best',num_labels=len(labels))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

#model = torch.load('Checkpoints/config.json')
#model = torch.hub.load('huggingface/transformers', 'XLMConfig', 'Checkpoints')
#config = AutoConfig.from_json_file('.Checkpoints/config.json')
#model = torch.hub.load('huggingface/pytorch-transformers', 'model', './tf_model/bert_tf_checkpoint.ckpt.index', from_tf=True, config=config)

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
        pred = outputs.logits.argmax(-1).squeeze().tolist()
        for id, label in zip(input_ids, pred):
          #print("Type id")
          #print(type(id))
          #print(id.shape)
          #print("Type pred")
          #print(type(pred))
          pred_ls = [id2label[label_one] for label_one in label]
          print("String:")
          print(len(label))
          #print(processor.tokenizer.decode(id), pred_ls)
          print("End")
          '''
          for i in range(len(pred)):
            print("- - - - -- - - - - -")
            print(processor.tokenizer.decode(id)[i])
            #print(label)
            print(id2label[label[i]]) 
            print("- - - - - - - - - - - ")
              #print(id2label[label]) for label in label.tolist() if label != -100
          '''
#debug_labels = [id2label[idx] for idx in preds_val]

print("END VAL")
print("START TEST")

import warnings
warnings.filterwarnings("ignore")