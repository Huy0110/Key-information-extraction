
# Key-information-extraction

# Requirement
### Run this to install transformer:
```bash
git clone -b modeling_layoutlmv2_v2 https://github.com/NielsRogge/transformers.git
```
```bash
pip install -q ./transformers
```
```bash
pip install git+https://github.com/huggingface/transformers
```
### Install:
- seqeval
- pyyaml==5.1
- torch==1.8.0+cu101 torchvision==0.9.0+cu101
- detectron2

# Dataset

- The example of original dataset: https://drive.google.com/drive/folders/12S7zZPQc0rkKgSEsxCnAlMpaI2oqKFGH?usp=sharing
- 3 forder train, val, test as in repo is the sample of result after process the original dataset 
- The formated dataset is similar to https://drive.google.com/drive/folders/1_r2rgPKBqqFmEFoNvz2lQGfIIfRALJ_W

# Run

## 1. Process with data
### Converted original dataset to 3 forder train,val,test
```bash
python data_process.py
```
### Make file .pkl for data  (In case of the input has token)
```bash
python generate_data.py
```

### Make file .pkl for data  (In case of the input has segment)
```bash
python new_gen_for_segment.py --path_train 'train/json' --path_val 'val/json' --path_test 'test/json'
```
### Visualise the data to verify
```bash
python visualize.py
```
### Note:
If you want to visualise the data, you need to remove the normalize step in gennerate_data.py

## 2. Train
```bash
bash script_train.sh
```

## 3. Test
```bash
python test.py --path_test 'Best'
```

## 4. Inference
### For the whole forder
```bash
python new_gennerate_for_inference.py --save_forder 'debug_pkl'
```

```bash
python new_get_tet_inference_result.py --path_check 'Best' --save_forder 'debug_pkl'
```

### For only one input
```bash
python new_gen_inference_for_one.py --path_image 'your image path' --path_json 'your json path' --path_save 'your path you want to save'
```
```bash
python new_inference_for_one.py --path_test 'your path(path_save in the previous command)' --path_check 'Best'
```


## 5. Save the Checkpoint to Drive
You can use rclone to do this task, for example:
```bash
bash script.sh
```
In addition, you can change the path or forder in script.sh to save the forder you want to your Drive

## References
https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/LayoutLMv2/CORD/Fine_tuning_LayoutLMv2ForTokenClassification_on_CORD.ipynb#scrollTo=Gq5CmIRZUy9O