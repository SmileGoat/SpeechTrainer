# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

import random
import sys
import os
import logging

import tensorflow as tf
import numpy as np

sys.path.insert(0, '../../')

from utils.kaldi_matrix import Matrix
from data import tfrecords_util as tf_util

#    def get_dim():
        #feature_dim_file = os.path.join(data_dir, 'feature_dim')
        #if os.path.exists(feature_dim_file):
            #with open(feature_dim_file, 'r') as f:
                #line = f.readlines()[-1]

        #feat_reader = Matrix()
        #with open(self.feats_scp, "r") as f:
            #line = f.readline().strip()
            #wav_id, file_obj_pos = line.split()
            #file_obj, pos = file_obj_pos.split(':')
            #pos = int(pos)
            #feat_data = feat_reader.read(file_obj, pos)
            #dim = feat_data.shape[0]

        #with open(feature_dim_file, 'w') as f:
            #f.write(str(dim))
        #return dim

class Feats2TFrecords(object):
    def __init__(self, data_dir, tfrecords_dir):
         # the dir includes: feats.scp
        # egs_range.1, egs_range.2, egs_range.3 ......
        self.data_dir = data_dir
        self.feats_scp = os.path.join(data_dir, "feats.scp")
        self.tfrecords_scp = os.path.join(data_dir, "tfrecords.scp")
#        feat_reader = CompressMatrix()
#        self.dim = self.get_dim()
        self.utt2spk = {}
        self.wav2feat = {}
        self.alignment_dict = {}
        self.tfrecords_dir = tfrecords_dir
        if not os.path.exists(self.tfrecords_dir):
            os.mkdir(self.tfrecords_dir)

        with open(os.path.join(data_dir, "feats.scp"), 'r') as f:
            for line in f:
                wav, feat = line.strip().split(" ")
                self.wav2feat[wav] = feat

        with open(os.path.join(data_dir, 'alignment.ark'), 'r') as f:        
            for line in f:
                line_array = line.rstrip().split()
                key = line_array[0]
                alignment = np.asarray([ int(x) for x in line_array[1:] ], np.int32)
                if key not in self.alignment_dict.keys():
                    self.alignment_dict[key] = alignment
                else :
                    logging.warning("duplicate alignments key " + key + "\n")

    def get_feature_and_idx(self, scp_line):
        line_list = scp_line.rstrip().split()
        wav_id = line_list[0]
        file_info_list = line_list[1].split(':')
        file_id = file_info_list[0]
        file_pos = int(file_info_list[1])
        return wav_id, file_id, file_pos

    def write_tfrecords(self):
        '''
        alignment: 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 4 4 4 4
        '''
        def write_tfrecord(tfrecord_file, filename, feature, alignment):
            tf_feature = {
                'wav_id':tf_util._bytes_feature(filename),
                'shape': tf_util._int64_feature(np.shape(feature)),
                'nnet_input': tf_util._float_feature(feature.flatten()),
                'nnet_output': tf_util._int64_feature(alignment),
            }
            tfwriter = tf.io.TFRecordWriter(tfrecord_file)
            tfwriter.write(tf_util.make_example(tf_feature))
        feat_reader = Matrix()
        tf_scp_writer = open(self.tfrecords_scp, 'w')
        with open(self.feats_scp, 'r') as feats_file:
            for line in feats_file:
                wav_id, file_ark, position = self.get_feature_and_idx(line)
                tfrecord_name = wav_id.split('/')[-1]
                tfrecord_name = tfrecord_name.replace('.wav','.tfrecord')
                feature = feat_reader.read(file_ark, position)
                tfrecord_file = os.path.join(self.tfrecords_dir, tfrecord_name)
                if wav_id not in self.alignment_dict.keys():
                    print('{0} is not in the alignment ark file'.format(wav_id))
                    continue
                alignment = self.alignment_dict[wav_id]
                tf_scp_item = wav_id + " " + tfrecord_file + "\n"
                tf_scp_writer.write(tf_scp_item)
                write_tfrecord(tfrecord_file, wav_id, feature, alignment)

if __name__ == '__main__':
   import argparse
   parser = argparse.ArgumentParser() 
   parser.add_argument('--data_dir', type=str, help='data dir: feats.scp alignment.ark')
   parser.add_argument('--tfrecords_dir', type=str, help='tfrecord dir')
   args = parser.parse_args()
   data_dir = args.data_dir
   tfrecords_dir = args.tfrecords_dir
   feat2tfrecords = Feats2TFrecords(data_dir, tfrecords_dir)
   feat2tfrecords.write_tfrecords()


