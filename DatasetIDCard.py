from os import listdir
from torch.utils.data import Dataset
import torch
from PIL import Image
import pandas as pd
from collections import Counter
from torch.utils.data import DataLoader
from transformers import LayoutLMv2ForTokenClassification, AdamW
from tqdm.notebook import tqdm
import json

class DatasetIDCard(Dataset):
    def __init__(self, annotations, image_dir, processor=None, max_length=512, label2id = None):
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
        self.label2id = label2id

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
        
        word_labels = [self.label2id[label] for label in word_labels]

        #word_labels = [self.label2id[label] for label in word_labels]
        # use processor to prepare everything
        encoded_inputs = self.processor(image, words, boxes=boxes, word_labels=word_labels, 
                                        padding="max_length", truncation=True, 
                                        return_tensors="pt", max_length = 512, return_token_type_ids = True)
        
        # remove batch dimension
        for k,v in encoded_inputs.items():
          encoded_inputs[k] = v.squeeze()
        #print(encoded_inputs.input_ids.shape)
        #print(encoded_inputs.attention_mask.shape)
        #print(encoded_inputs.token_type_ids.shape)
        #print(encoded_inputs.bbox.shape)
        #print(encoded_inputs.image.shape)
        #print(encoded_inputs.labels.shape)

        assert encoded_inputs.input_ids.shape == torch.Size([512])
        assert encoded_inputs.attention_mask.shape == torch.Size([512])
        assert encoded_inputs.token_type_ids.shape == torch.Size([512])
        assert encoded_inputs.bbox.shape == torch.Size([512, 4])
        assert encoded_inputs.image.shape == torch.Size([3, 224, 224])
        assert encoded_inputs.labels.shape == torch.Size([512]) 
      
        return encoded_inputs