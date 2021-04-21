import sys
import six

import tensorflow as tf

from models.speaker_verification.tdnn import tdnn_kaldi
from models.speaker_verification.tdnn import tdnn
from models.speaker_verification.tdnn import lstm
from data.pipeline import make_pipeline

def make_graph(pipeline, lrate, l2_regularize_factor, is_training = True):
    nnet_input = pipeline['nnet_input']
    nnet_output = pipeline['nnet_output']
    graph = dict()
    optimizer = tf.train.GradientDescentOptimizer(lrate, name='optimizer')
    logits, endpoints = tdnn_kaldi(nnet_input, l2_regularize_factor)
    loss = tf.losses.sparse_softmax_cross_entropy(nnet_output, logits)
    grads = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads)
    embedding = endpoints['tdnn6_dense']

    graph['loss_op'] = loss
    graph['nnet_input'] = nnet_input
    graph['nnet_target'] = nnet_output
    graph['logits'] = logits
    graph['lrate'] = lrate
    graph['train_op'] = train_op
    graph['embedding'] = embedding
    return graph


def make_validation_graph(pipeline):
    nnet_input = pipeline['nnet_input']
    nnet_output = pipeline['nnet_output']
    graph = dict()
    logits, endpoints = tdnn_kaldi(nnet_input)
    loss = tf.losses.sparse_softmax_cross_entropy(nnet_output, logits)
    prediction = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(nnet_output, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    graph['loss'] = loss
    graph['accuracy'] = acc
    return graph

def make_inference_graph(nnet_input):
    graph = dict()
    _, endpoints = tdnn_kaldi(nnet_input)
    graph['embedding'] = endpoints
    return graph

