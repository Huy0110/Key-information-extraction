import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import LayoutLMv2ForTokenClassification, AdamW
import torch
from tqdm.notebook import tqdm
from os import listdir
from PIL import Image
from clearml import Task
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs")
task = Task.init(project_name='ID Card', task_name='test model')



def replace_elem(elem):
  try:
    return replacing_labels[elem]
  except KeyError:
    return elem

def replace_list(ls):
  return [replace_elem(elem) for elem in ls]

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

train = pd.read_pickle('train.pkl')
val = pd.read_pickle('dev.pkl')
test = pd.read_pickle('test.pkl')
all_labels = [item for sublist in train[1] for item in sublist] + [item for sublist in val[1] for item in sublist] + [item for sublist in test[1] for item in sublist]
#print(Counter(all_labels))
replacing_labels = {'title_en': 'title', 'chxh_2_en':'O','key_ngon_tro_trai_en':'O','key_ngon_tro_phai_en':'O','value_nguoi_cap':'O','key_ngon_tro_trai':'O','cnxh_1_en': 'O','chxh_1_en':'O','ignore_1_en': 'O', 'ignore_2_en':'O', 'noi_cap_1':'O', 'noi_cap_2': 'O', 'driver_license_bo_gtvt_1': 'O', 'driver_license_bo_gtvt_2': 'O', 'dau_vet_1': 'dau_vet', 'dau_vet_2': 'dau_vet', 'ignore_1': 'O', 'ignore_2': 'O', 'chxh_1': 'O', 'chxh_2': 'O', 'key_ho_ten_khac':'O', 'level_nguoi_cap_1': 'O', 'level_nguoi_cap_2': 'O', 'key_hang_cap': 'O', 'nguoi_cap': 'O', 'hang_cap': 'O','van_tay_phai':'O','van_tay_trai':'O'}

train[1] = [replace_list(ls) for ls in train[1]]
val[1] = [replace_list(ls) for ls in val[1]]
test[1] = [replace_list(ls) for ls in test[1]]
all_labels = [item for sublist in train[1] for item in sublist] + [item for sublist in val[1] for item in sublist] + [item for sublist in test[1] for item in sublist]
labels = list(set(all_labels))
labels = sorted(labels)
#print(labels)

label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for idx, label in enumerate(labels)}
print(label2id)
#print(all_labels)
#print(id2label)

class Dataset(Dataset):

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

        #assert len(words) == len(boxes) == len(word_labels)

        """ print(len(words))
        print(len(boxes))
        print(len(word_labels))
        print(self.image_dir) """
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

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4)


model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-base-uncased',
                                                                      num_labels=len(labels))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

global_step = 0
eval_freq = 250
latest_freq = 500
display_freq = 100
loss_best = 100
num_train_epochs = 4


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
            model.save_pretrained("Checkpoints/Best")
            print("Saving the best model")
          model.train()
        
        if global_step % latest_freq == 0 and global_step >= latest_freq:
          model.save_pretrained("Checkpoints/Latest")
          print("Saving the latest model")

        model.save.pretrained("Checkpoints/Epoch")
        print("Saving the latest epoch model")

        loss.backward()
        optimizer.step()
        global_step += 1

#model.save_pretrained("Checkpoints")