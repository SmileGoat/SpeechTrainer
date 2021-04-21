# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)


import argparse
import sys
import os

def get_writers(file, egs_path):
    ark_idx = []
    with open(file, 'r') as f:
        for line in f:
            line_arr = line.strip().split()
            ark_id = line_arr[2]
            if ark_id not in ark_idx:
                ark_idx.append(ark_id)
    egs_writers = []
    ark_idx.sort()
    print(ark_idx)
    for ark_id in ark_idx:
        egs_name = 'egs_range.' + ark_id
        fd = open(os.path.join(egs_path, egs_name), 'w')
        egs_writers.append(fd)
    return egs_writers

def main(argv):
    range_file = argv[1]
    egs_path = argv[2]
    egs_writers = get_writers(range_file, egs_path)
    with open(range_file) as f:
        for line in f:
            ark_ref = int(line.strip().split()[1])
            egs_writers[ark_ref].write(line)

    for writer in egs_writers:
        writer.close()

if __name__ == "__main__":
    main(sys.argv)

