# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

import sys
import os
import threading

from nnet.graph import make_graph
from nnet.graph import make_inference_graph
from data.pipeline import make_pipeline
from utils.gpu import average_models
from utils.kaldi_compressed_matrix import CompressMatrix
from utils.kaldi_matrix import Matrix
import numpy as np
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
tf.logging.set_verbosity(tf.logging.INFO)

def predict_embedding(feats_scp, ark_file, model_eval, cpu_id):
    # input feature chunks is [batch, length, dim]
    device_id="/cpu:" + str(cpu_id)
    feature_dim = 20
    matrix_writer = Matrix(ark_file)
    nnet_input = tf.placeholder(tf.float32, shape=(None, None, feature_dim))
    #sess_config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=True)
    sess_config = tf.ConfigProto(intra_op_parallelism_threads=1,
                                 inter_op_parallelism_threads=1,
                                 device_count={'CPU': 50},
                                 allow_soft_placement=True)

    sess = tf.Session(config = sess_config)
    with tf.device(device_id):
        graph = make_inference_graph(nnet_input)
    saver = tf.train.Saver()
    saver.restore(sess, model_eval)
    for index, (key, feature_chunks, feature_length) in enumerate(make_feature_chunks_kaldi(feats_scp, 41, 40)):
        tf.logging.info("[INFO] Key %s " % (key))
        embedding_opt = graph['embedding']
        embeddings = sess.run(embedding_opt, feed_dict={nnet_input:np.array(feature_chunks[:-1], dtype=np.float32)})
#        embeddings_last = sess.run(embedding_opt, feed_dict={nnet_input:np.array(feature_chunks[-1], dtype=np.float32)})
        embedding = merge_embeddings(embeddings['tdnn6_dense'])
        matrix_writer.write_vector(key, embedding)
    sess.close()
    return embeddings


def predict_embedding_test():
    # input feature chunks is [batch, length, dim]
    feature_dim = 20
    feats_scp = "~/workspace/tf_models/speaker_verification/test_tfrecords/inference_feats.scp" 
    ark_file = "~/workspace/tf_models/speaker_verification/test_tfrecords/inference_embedding.ark" 
    model_name="~/workspace/tf_models/speaker_verification/models/final.raw"
    matrix_writer = Matrix(ark_file)
    nnet_input = tf.placeholder(tf.float32, shape=(None, None, feature_dim))
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config = sess_config)
    graph = make_inference_graph(nnet_input)
    saver = tf.train.Saver()
    saver.restore(sess, model_name)
    for index, (key, feature_chunks, feature_length) in enumerate(make_feature_chunks_kaldi(feats_scp, 41, 40)):
        print(len(feature_chunks))
        embedding_opt = graph['embedding']
        embeddings = sess.run(embedding_opt, feed_dict={nnet_input:np.array(feature_chunks[:-1], dtype=np.float32)})
        embedding = merge_embeddings(embeddings['tdnn6_dense'])
        matrix_writer.write_vector(key, embedding)
    sess.close()
    return embedding

def merge_embeddings(embeddings:list):
    embedding = np.sum(embeddings, axis=0) / len(embeddings)
    return embedding

def make_feature_chunks(feat_scp, min_chunk_size, chunk_size):
    matrix_reader = Matrix()
    for index, (key, feature) in enumerate(matrix_reader.read_mat_scp(feat_scp)):
        if feature.shape[0] < chunk_size:
            tf.logging.info("[INFO] Key %s length too short, %d < %d, skip." % (key, feature.shape[0], min_chunk_size))
            continue
        if feature.shape[0] > chunk_size:
            feature_array = []
            feature_length = []
            num_chunks = int(np.ceil(float(feature.shape[0] - chunk_size) / (chunk_size / 2))) + 1
            tf.logging.info("[INFO] Key %s length %d > %d, split to %d segments." % (key, feature.shape[0], chunk_size, num_chunks))
            for i in range(num_chunks):
                start = int(i * (chunk_size / 2))
                this_chunk_size = chunk_size if feature.shape[0] - start > chunk_size else feature.shape[0] - start
                feature_length.append(this_chunk_size)
                feature_array.append(feature[start:start+this_chunk_size])
            # feature_array is [num_chunk, chunk_size, feature_dim]
            yield key, feature_array, feature_length
#            feature_length = np.expand_dims(np.array(feature_length), axis=1)

def make_feature_chunks_kaldi(feat_scp, min_chunk_size, chunk_size):
    matrix_reader = Matrix()
    for index, (key, feature) in enumerate(matrix_reader.read_mat_scp(feat_scp)):
        if feature.shape[0] < min_chunk_size:
            tf.logging.info("[INFO] Key %s length too short, %d < %d, skip." % (key, feature.shape[0], min_chunk_size))
            continue
        if feature.shape[0] < chunk_size:
            chunk_size = feature.shape[0]
            continue
        feature_array = []
        feature_length = []
        num_chunks = int(np.ceil(float(feature.shape[0] / chunk_size)))
        tf.logging.info("[INFO] Key %s length %d > %d, split to %d segments." % (key, feature.shape[0], chunk_size, num_chunks))
        if (feature.shape[1] != 20):
            tf.logging.info("[INFO] Key %s col %d, dim:  %d, wrong data" % (key, feature.shape[0], feature.shape[1]))
            continue
        for i in range(num_chunks):
            start = i * chunk_size
            this_chunk_size = chunk_size if feature.shape[0] - start > chunk_size else feature.shape[0] - start
            feature_length.append(this_chunk_size)
            feature_array.append(feature[start:start+this_chunk_size])
        yield key, feature_array, feature_length

def main(argv):
    feats_scp = argv[1]
    model_eval = argv[2]
    ark_file = argv[3]
    cpu_id = argv[4]
    predict_embedding(feats_scp, ark_file, model_eval, cpu_id)


if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv)
-
