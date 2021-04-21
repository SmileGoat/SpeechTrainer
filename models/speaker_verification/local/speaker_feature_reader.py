# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

import random
import sys
import os
import sys
import threading

import tensorflow as tf
import numpy as np

from utils.file_utils import get_files_in_dir
from utils.kaldi_compressed_matrix import CompressMatrix
from data import tfrecords_util as tf_util

class FeatureReader(object):
    def __init__(self, data_dir, batch_size, tfrecords_dir, njobs):
        # the dir includes: feats.scp
        # egs_range.1, egs_range.2, egs_range.3 ......
        self.data_dir = data_dir
        self.feats_scp = os.path.join(data_dir, "feats.scp")
#        feat_reader = CompressMatrix()
        self.dim = self.get_dim()
        self.utt2spk = {}
        self.spk2int = {}
        self.wav2feat = {}
        self.batch_size = batch_size
        #self.range_files = get_files_in_dir(data_dir, "egs_range.*")
        self.tfrecords_dir = tfrecords_dir
        self.njobs = njobs
        if not os.path.exists(self.tfrecords_dir):
            os.mkdir(self.tfrecords_dir)

        with open(os.path.join(data_dir, "feats.scp"), 'r') as f:
            for line in f:
                wav, feat = line.strip().split(" ")
                self.wav2feat[wav] = feat

    def scp2tfrecords_sub(self, prefix:str = ''):
        match_str = prefix + "egs_range.*"
        range_files = get_files_in_dir(self.data_dir, match_str)
        self.scp2tfrecords_multi_thread(range_files, prefix)

    def scp2tfrecords_multi_thread(self, range_files, prefix):
        def sub_run(range_file, prefix:str = ''):
           self.write_batch_tfrecords(range_file, prefix)

        range_len = len(range_files)
        threads = []
        if range_len < self.njobs:
            for idx in range(range_len):
                threads.append(threading.Thread(target=sub_run, args=(range_files[idx], prefix)))
                threads[idx].start()
            for thread in threads:
                thread.join()
        else:
            for idx in range(range_len):
                print(idx)
                threads.append(threading.Thread(target=sub_run, args=(range_files[idx], prefix)))
                print("xxxx")
                threads[idx].start()
                if ((idx + 1) % self.njobs) == 0:
                    print("in join process")
                    for thread in threads:
                        thread.join()
                    threads.clear()
                print(idx+1)
            for thread in threads:
                thread.join()

    
    def scp2tfrecords(self):
#        self.scp2tfrecords_sub()
        self.scp2tfrecords_sub("valid_")
        #self.scp2tfrecords_sub("train_subset_")

    def get_dim(self):
        feat_reader = CompressMatrix()
        with open(self.feats_scp, "r") as f:
            line = f.readline().strip()
            wav_id, file_obj_pos = line.split()
            file_obj, pos = file_obj_pos.split(':')
            pos = int(pos)
            feat_reader.read(file_obj, pos)
            feat_data = feat_reader.numpy()
            dim = feat_data.shape[1]

        with open(os.path.join(data_dir, 'feature_dim'), 'w') as f:
            f.write(str(dim))
        return dim 

    def write_batch_tfrecords(self, eg_range_file:str, prefix:str = ''):
        '''
        eg in range :
        wav_id base_ark relative_ark begin_frame frame_num spk_id
        20884965_16 0 1 20 113 0 
        20884965_28 0 1 14 113 0            
        '''
        def write_tfrecord(file_name, batch_features, length, batch_spk):
            feature = {
                # batch_size, length, dim
                'length' : tf_util._int64_feature(length),
                'nnet_input' : tf_util._float_feature(batch_features.flatten()),
                'nnet_output' : tf_util._int64_feature(batch_spk),
            }
            tfwriter = tf.io.TFRecordWriter(file_name)
            tfwriter.write(tf_util.make_example(feature))

        tfrecords_dir = self.tfrecords_dir 
        batch_idx = 0
        #wav_id, _, ark_id, frame_start, batch_length, spk_id = eg_range[0].split()
        batch_length = 40

        # print(self.batch_size, batch_length, self.dim)
        batch_features = np.zeros((self.batch_size, batch_length, self.dim), dtype=np.float32)
        batch_spkid = np.zeros(self.batch_size, dtype=np.int32)
        feat_reader = CompressMatrix()
        tf_idx = 0
        with open(eg_range_file, 'r') as eg_range:
            for eg in eg_range:
                wav_id, _, ark_id, frame_start, length, spk_id  = eg.strip().split()
                frame_start = int(frame_start)
                length = int(length)
                spk_id = int(spk_id)
                ark_file, position = self.get_feature_and_idx(wav_id)
                feat_reader.read(ark_file, position)
                feature = feat_reader.numpy() 
                feat_chunk = feature[frame_start:frame_start+length]
    #            print(ark_file)
    #            print(feat_chunk)
                batch_features[batch_idx,:,:] = feat_chunk
                batch_spkid[batch_idx] = spk_id
                batch_idx += 1
                if batch_idx == self.batch_size:
    #                file_prefix = "eg" if prefix == '' else "eg_"
                    file_name = "eg_" + prefix + str(ark_id) + "_batch_"+ str(tf_idx) + ".tfrecord"
                    file_name = os.path.join(tfrecords_dir, file_name)
                    print(file_name)
                    write_tfrecord(file_name, batch_features, length, batch_spkid)
                    batch_idx = 0
                    tf_idx += 1

    def get_feature_and_idx(self, wav_id):
        feat_str = self.wav2feat[wav_id]
        file_id, file_pos = feat_str.split(':')
        file_pos = int(file_pos)
        return file_id, file_pos

    def shuffle_egs(self, buffer_size:int = None, seed=124, range_egs:list = None) -> list:
        if buffer_size is None:
            #shuffle all egs
            range_egs = random.shuffle(range_egs)
            return range_egs
        egs_shuffle = []
        egs = {}
        random.seed(seed)
        for eg in range_egs:
            index = random.randint(0, buffer_size - 1)
            if index not in egs:
                egs_shuffle.append(eg)
                egs[index] = eg
            else:
                egs_shuffle.append(eg)
                egs[index] = eg
        return egs_shuffle

import argparse

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--dir', type=str, help='dir')
   parser.add_argument('--tfrecords_dir', type=str, help='tfrecord dir')
   parser.add_argument('--batch_size', type=int, help='batch size')
   parser.add_argument('--njobs', type=int, help=10)
   args = parser.parse_args()
   data_dir = args.dir
   tfrecords_dir = args.tfrecords_dir
   batch_size = args.batch_size
   njobs = args.njobs
   sv_feature = FeatureReader(data_dir, batch_size, tfrecords_dir, njobs)
   sv_feature.scp2tfrecords()

