# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)


import wave
import os
import sys

pcm_file = sys.argv[1]
wav_file = sys.argv[2]


#wav_name = pcm_file.split('.')[0]
with open(pcm_file, 'rb') as pcmfile:
   pcmdata = pcmfile.read()
   with wave.open(wav_file, 'wb') as wavfile:
      wavfile.setparams((1, 2, 16000, 0, 'NONE', 'NONE'))
      wavfile.writeframes(pcmdata)


