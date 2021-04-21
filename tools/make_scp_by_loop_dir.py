#! /usr/bin/env python2
# -*- coding: utf-8 -*-

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
  root_dir = sys.argv[1]
  wave_scp = sys.argv[2]
#  waves_in_dir = os.listdir(wave_dir)
  write_file = open(wave_scp, 'w')

  for subdir, dirs, files in os.walk(root_dir):
    print subdir + " "+ str(len(files))
    for file in files:
      try:
#       realpath_wav = wave_dir + '/' + wave_name
        realpath_wav = subdir + os.sep + file
        wav_id = wave.open(realpath_wav,'r')
      except wave.Error:
        print realpath_wav + " open error"
        continue
      frames = wav_id.getnframes()
      if frames == 0 :
        print realpath_wav + " is empty file"
        continue
      frame_rate = wav_id.getframerate()
      if frame_rate == 16000 :
        write_file.write(realpath_wav + '\n')


