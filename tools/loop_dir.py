# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)


import os
import sys
import wave
import shutil

def Usage():
    print "Usage: %s src_dir wav_scp \n attention: dir input like /tmp/src_dir, dont like /tmp/src_dir/ " %(sys.argv[0],)
    sys.exit(-1)

if __name__ == '__main__':
  if len(sys.argv) != 3:
     Usage()
  scp_file = sys.argv[1]
  wave_scp = sys.argv[2]
  read_file = open(scp_file, 'r') 
  dir_lines = read_file.readlines()
  write_file = open(wave_scp, 'w')

  for wave_dir in dir_lines: 
#       realpath_wav = wave_dir + '/' + wave_name
#        realpath_wav = subdir + os.sep + file
      wave_dir = wave_dir.strip()
      waves_dir = os.listdir(wave_dir)
      for wave_name in waves_dir: 
          write_file.write(wave_dir + '/' + wave_name + '\n')


