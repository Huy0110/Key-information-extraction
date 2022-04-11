import subprocess
import os
from tqdm.notebook import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path_check', required=True, type=str)
parser.add_argument('--save_forder', required=True, type=str)

path_check = vars(parser.parse_args())['path_check']
save_forder = vars(parser.parse_args())['save_forder']
list_dir_debug = save_forder
dir_forder = os.listdir(list_dir_debug)
dir_forder = sorted(dir_forder)
for list_dir in dir_forder:
    list_dir = os.path.join(save_forder, list_dir)
    #print(list_dir)
    cmd = ["python", "new_test_inference.py", "--path_test", list_dir, "--path_check", path_check]
    subprocess.run(cmd)