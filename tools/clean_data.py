# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)


import os
import sys
import wave
import shutil
import librosa

def Usage():
    print('Usage: {0} src_dir dst_dir \n attention: dir input like /tmp/src_dir, dont like /tmp/src_dir/ '.format(sys.argv[0]))
    sys.exit(-1)

if __name__ == '__main__':
  if len(sys.argv) != 3:
     Usage()
  wave_dir = sys.argv[1]
  wave_store_dir = sys.argv[2]
  if not os.path.exists(wave_store_dir):
      os.makedirs(wave_store_dir)
  waves_in_dir = os.listdir(wave_dir)
  base_dir = os.path.basename(wave_dir)
   
  for wave_name in waves_in_dir:
    try:
       realpath_wav = wave_dir + '/' + wave_name
       wav_id = wave.open(realpath_wav,'r')
    except :
       print(wave_name + " open error")
       continue
    frames = wav_id.getnframes()
    if frames == 0 :
       print(wave_name + " is empty file")
       continue
    frame_rate = wav_id.getframerate()
    if frame_rate == 16000 :
       dst_wav = wave_store_dir + "/" + wave_name
       shutil.copyfile(realpath_wav, dst_wav)
    elif frame_rate > 16000 :
       print(wave_name + ": resampling to 16k")
       audio_, sr = librosa.load(realpath_wav, frame_rate)
       audio_16k = librosa.resample(y=audio_, orig_sr=sr, target_sr=16000)
       dst_wav = wave_store_dir + "/" + wave_name
       librosa.output.write_wav(dst_wav, audio_16k, 16000)


