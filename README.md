# Key-information-extraction

# Requirement
### Run this to install transformer:
```bash
git clone -b modeling_layoutlmv2_v2 https://github.com/NielsRogge/transformers.git
pip install -q ./transformers
```
### Install:
- seqeval
- pyyaml==5.1
- torch==1.8.0+cu101 torchvision==0.9.0+cu101
- detectron2

# Dataset

- The example of original datase has 3 forders: Train, Val, Test as in this repo
- The formated dataset is similar to https://drive.google.com/drive/folders/1_r2rgPKBqqFmEFoNvz2lQGfIIfRALJ_W

# Run
## Converted original dataset to the formated dataset
```bash
python data_process.py
```
## Make file .pkl for data
```bash
python gennerate_data.py
```
## Visualise the data to verify
```bash
python visualise.py
```
### Note:
If you want to visualise the data, you need to remove the normalize step in gennerate_data.py

## Train
```bash
python train.py
```

## Test
```bash
python test.py
```
