# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

import sys
import os
import threading
import six

from nnet.graph import make_graph
from nnet.graph import make_inference_graph
from data.pipeline import make_pipeline
from utils.gpu import average_models
from utils.kaldi_compressed_matrix import CompressMatrix
from utils.kaldi_matrix import Matrix
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="1"

def predict_embedding(feats_scp, ark_file, model_eval):
    # input feature chunks is [batch, length, dim]
    feature_dim = 20
    matrix_writer = Matrix(ark_file)
    nnet_input = tf.placeholder(tf.float32, shape=(None, None, feature_dim))
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config = sess_config)
    graph = make_inference_graph(nnet_input)
    saver = tf.train.Saver()
    saver.restore(sess, model_eval)
    for index, (key, feature_chunks, feature_length) in enumerate(make_feature_chunks_kaldi(feats_scp, 40, 40)):
        embedding_opt = graph['embedding']
        embeddings = sess.run(embedding_opt, feed_dict={nnet_input:np.array(feature_chunks[:-1], dtype=np.float32)})
        matrix_writer.write(key, embeddings['tdnn6_dense'])
    sess.close()
    return embeddings


def predict_embedding_test():
    # input feature chunks is [batch, length, dim]
    feature_dim = 20
    feats_scp = "~/speaker_verification/test_tfrecords/inference_feats.scp" 
    ark_file = "~/speaker_verification/test_tfrecords/inference_embedding.ark" 
    matrix_writer = Matrix(ark_file)
    nnet_input = tf.placeholder(tf.float32, shape=(None, None, feature_dim))
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config = sess_config)
    graph = make_inference_graph(nnet_input)
    saver = tf.train.Saver()
    model_path="~/model/sv_ckpt_average.mdl"
    saver.restore(sess, model_path)
    for index, (key, feature_chunks, feature_length) in enumerate(make_feature_chunks_kaldi(feats_scp, 40, 40)):
        embedding_opt = graph['embedding']
        print('feature_chunks shape :', len(feature_chunks))
        # the chunks [chunk_size, dim]
        embeddings = sess.run(embedding_opt, feed_dict={nnet_input:np.array(feature_chunks[:-1], dtype=np.float32)})
        print("embeddings shape: ", len(embeddings['tdnn6_dense']))
        embedding = merge_embeddings(embeddings['tdnn6_dense'])
        matrix_writer.write_vector(key, embedding)
    sess.close()
    return embeddings

def merge_embeddings(embeddings:list):
    embedding = np.sum(embeddings, axis=0) / len(embeddings)
    return embedding

def make_feature_chunks_gap(feat_scp, min_chunk_size, chunk_size):
    matrix_reader = CompressMatrix()
    for index, (key, feature) in enumerate(matrix_reader.read_compressed_scp(feat_scp)):
        if feature.shape[0] < min_chunk_size:
            tf.logging.info("[INFO] Key %s length too short, %d < %d, skip." % (key, feature.shape[0], min_chunk_size))
            continue
        if feature.shape[0] > chunk_size:
            feature_array = []
            feature_length = []
            num_chunks = int(np.ceil(float(feature.shape[0] - chunk_size) / (chunk_size / 2))) + 1
            tf.logging.info("[INFO] Key %s length %d > %d, split to %d segments." % (key, feature.shape[0], chunk_size, num_chunks))
            for i in range(num_chunks):
                start = i * (chunk_size / 2)
                this_chunk_size = chunk_size if feature.shape[0] - start > chunk_size else feature.shape[0] - start
                feature_length.append(this_chunk_size)
                feature_array.append(feature[start:start+this_chunk_size])
            yield key, feature_array, feature_length

def make_feature_chunks_kaldi(feat_scp, min_chunk_size, chunk_size):
    matrix_reader = CompressMatrix()
    for index, (key, feature) in enumerate(matrix_reader.read_compressed_scp(feat_scp)):
        if feature.shape[0] < min_chunk_size:
            tf.logging.info("[INFO] Key %s length too short, %d < %d, skip." % (key, feature.shape[0], min_chunk_size))
            continue
        if feature.shape[0] < chunk_size:
            chunk_size = feature.shape[0]
        feature_array = []
        feature_length = []
        num_chunks = int(np.ceil(float(feature.shape[0] / chunk_size)))
        tf.logging.info("[INFO] Key %s length %d > %d, split to %d segments." % (key, feature.shape[0], chunk_size, num_chunks))
        for i in range(num_chunks):
            start = i * chunk_size 
            this_chunk_size = chunk_size if feature.shape[0] - start > chunk_size else feature.shape[0] - start
            feature_length.append(this_chunk_size)
            feature_array.append(feature[start:start+this_chunk_size])
        yield key, feature_array, feature_length

def Usage():
    print ("usage:%s feats model ark" %(sys.argv[0],))
    exit(-1)

def main(argv):
    predict_embedding_test()
    exit()
    if len(argv) != 4:
        Usage()
    feats_scp = argv[1] 
    model_eval = argv[2]
    ark_file = argv[3] 
    predict_embedding(feats_scp, ark_file, model_eval)

if __name__ == '__main__':
    main(sys.argv)
