# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)


import os
import sys
import wave
import shutil

spkutt_id = sys.argv[1]
wav_scp = sys.argv[2]

wav_file = open(wav_scp, 'r')
wav_scp_lines = wav_file.readlines()

id_dict = {}
for line in wav_scp_lines:
    line_arr = line.strip().split()
    id_dict[line_arr[0]] = line_arr[1]

spkutt_file = open(spkutt_id, 'r')
spkutt_lines = spkutt_file.readlines()

for line in spkutt_lines:
    line_arr = line.strip().split() 
    new_line = line_arr[0]
    for x in range(1, len(line_arr)):
        new_line += " " + id_dict[line_arr[x]]
    print new_line


