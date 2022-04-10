import subprocess
import os
from tqdm.notebook import tqdm
list_dir_debug = 'debug_pkl_real'
dir_forder = os.listdir(list_dir_debug)
dir_forder = sorted(dir_forder)
for list_dir in dir_forder:
    list_dir = os.path.join('debug_pkl_real', list_dir)
    #print(list_dir)
    cmd = ["python", "new_test_inference.py", "--path_test", list_dir]
    subprocess.run(cmd)