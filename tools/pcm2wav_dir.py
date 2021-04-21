#! /usr/bin/env python2
# -*- coding: utf-8 -*-

import wave
import os
import sys

directory = sys.argv[1] 
directory_wav = sys.argv[2]

direcotory = os.fsencode(directory)
for file in os.listdir(directory):
  filename = os.fsdecode(file)
  print(filename)
  #if filename.endwith(".pcm"):
  pcm_file = os.path.join(directory, filename)
  wav_file = os.path.join(directory_wav, filename)
  wav_name = wav_file.split('.')[0]
  print(wav_name)
  with open(pcm_file, 'rb') as pcmfile:
    pcmdata = pcmfile.read()
  with wave.open(wav_name + '.wav', 'wb') as wavfile:
    wavfile.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
    wavfile.writeframes(pcmdata)


