# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

import tensorflow as tf
import os
import sys

from nnet import DNN

def splice(feature, frame_size, feature_dim, left_context, right_context):
    #the shape of input feature is [frame, feature_dim]
    #the shape of output feature is [frame, (left_context+right_context+1)*feature_dim]
    spliced_feature = []
    first_frame = tf.slice(feature, [0, 0], [1, feature_dim])
    last_frame = tf.slice(feature, [frame_size - 1, 0], [1, feature_dim])
    prefix = tf.tile(first_frame, [right_context, 1]) # [right_context, dim]
    postfix = tf.tile(last_frame, [left_context, 1]) 
    new_feature = tf.concat([prefix, feature, postfix], 0) #[righ_context + frame + left_context, dim]
    for i in range(left_context + 1 + right_context):
        elem = tf.slice(new_feature, [i, 0], [frame_size, feature_dim]) 
        spliced_feature.append(elem)
    return tf.concat(spliced_feature, 1)

def read_dataset_from_tfrecords(tfrecords_list, left_context=5, right_context=10):
    res = dict()
    def _parse_feature(example_proto):
        feature_description = {
            'wav_id': tf.io.FixedLenFeature([], tf.string),
            'shape': tf.io.FixedLenFeature([2], tf.int64),
            'nnet_input': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'nnet_output': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        }
        features = tf.io.parse_single_example(example_proto, feature_description)
        features['nnet_input'] = tf.reshape(features['nnet_input'], features['shape'])
        feature_dim = features['shape'][1]
        frame_size = features['shape'][0]
        # the output is [frame, (left_context+right_context+1)*feature_dim]
        res['nnet_input'] = splice(features['nnet_input'], frame_size, feature_dim, left_context, right_context)
        res['nnet_output'] = features['nnet_output']
        return res 
    with tf.device('/cpu:0'):
        raw_dataset = tf.data.TFRecordDataset(tfrecords_list).map(_parse_feature, num_parallel_calls=32)
        raw_dataset = raw_dataset.flat_map(lambda x : tf.data.Dataset.from_tensor_slices(x))
        return raw_dataset

def test_read(tfrecords_list):
    dataset = read_dataset_from_tfrecords(tfrecords_list)
    for feature in dataset.take(1):
        print(feature['nnet_input'])
        print(feature['nnet_output'])

import time

def cross_entropy_loss(x, y):
    y = tf.cast(y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    return tf.reduce_mean(loss)

def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)
