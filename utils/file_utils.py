# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

import os
import re

def get_files_in_dir(data_dir, suffix=''):
    def rank_keys(item):
        return (len(item), item)

    all_files = []
    for folder, subfolders, files in os.walk(data_dir):
        if len(files) > 0:
            files.sort(key=rank_keys)
            for f in files:
                file_abs_path = os.path.join(os.path.abspath(folder), f)
                if len(suffix) > 0 and not file_abs_path.endswith(suffix):
                    continue	# filter out other file types
                all_files.append(file_abs_path)

    return all_files

def get_reg_files_in_dir(data_dir, reg=''):
    def rank_keys(item):
        return (len(item), item)

    all_files = []
    pattern = re.compile(reg)
    for folder, subfolders, files in os.walk(data_dir):
        if len(files) > 0:
            files.sort(key=rank_keys)
            for f in files:
                file_abs_path = os.path.join(os.path.abspath(folder), f)
                if len(reg) > 0 and not bool(pattern.match(f)):
                    continue	# filter out other file types
                all_files.append(file_abs_path)

    return all_files



def read_first_and_last_line(input_file):
    with open(input_file, 'r') as f:
        first_line = f.readline()
        offset = -50
        while True:
            f.seek(offset, 2)
            lines = f.readlines()
            if len(lines) >= 2:
                last_line = lines[-1]
                break
            offset *= 2
    return first_line, last_line
