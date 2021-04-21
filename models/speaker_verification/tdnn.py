# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

import tensorflow as tf
import collections
from nnet.layers.pooling import statistics_pooling 
from nnet.layers.attention import attention

num_classes=1000

def tdnn_kaldi(features, is_training=None):
    """Build a TDNN network.
    The structure is the tdnn in Kaldi.
    """
    tf.logging.info("Build a standard TDNN network.")
    relu = tf.nn.relu

    endpoints = collections.OrderedDict()
    with tf.variable_scope("tdnn", reuse=tf.AUTO_REUSE):
        # the orignal feature is [batch_size, length, dim]

        # Layer 1: [-2,-1,0,1,2] --> [b, l-4, 512]
        # conv2d + batchnorm + relu
        features = tf.layers.conv1d(features,
                                512,
                                5,
                                activation=None,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                name='tdnn1_conv')
        endpoints["tdnn1_conv"] = features
        features = relu(features, name='tdnn1_relu')
        features = tf.layers.batch_normalization(features,
                                                 momentum=0.99,
                                                 training=is_training,
                                                 name="tdnn1_bn")
        endpoints["tdnn1_bn"] = features
        endpoints["tdnn1_relu"] = features

        # Layer 2: [-2,  0,  2] --> [b , l-4, 512]
        # conv2d + batchnorm + relu
        # This is slightly different with Kaldi which use dilation convolution
        features = tf.layers.conv1d(features,
                                    512,
                                    5,
                                  #  dilation_rate=2,
                                    activation=None,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                    name='tdnn2_conv')
        endpoints["tdnn2_conv"] = features
        features = relu(features, name='tdnn2_relu')
        features = tf.layers.batch_normalization(features,
                                                 momentum=0.99,
                                                 training=is_training,
                                                 name="tdnn2_bn")
        endpoints["tdnn2_bn"] = features
        endpoints["tdnn2_relu"] = features

        # Layer 3: [-3, 0, 3] --> [b,  l-6, 512]
        # conv2d + batchnorm + relu
        # Still, use a non-dilation one
        features = tf.layers.conv1d(features,
                                    512,
                                    7,
                                  #  dilation_rate=3,
                                    activation=None,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                    name='tdnn3_conv')
        endpoints["tdnn3_conv"] = features
        features = relu(features, name='tdnn3_relu')
        features = tf.layers.batch_normalization(features,
                                                 momentum=0.99,
                                                 training=is_training,
                                                 name="tdnn3_bn")
        endpoints["tdnn3_bn"] = features
        endpoints["tdnn3_relu"] = features

        # Convert to [b, l, 512]
        # Layer 4: [b, l, 512] --> [b, l, 512]
        features = tf.layers.dense(features,
                                   512,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                   name="tdnn4_dense")
        endpoints["tdnn4_dense"] = features
        features = relu(features, name='tdnn4_relu')
        features = tf.layers.batch_normalization(features,
                                                 momentum=0.99,
                                                 training=is_training,
                                                 name="tdnn4_bn")
        endpoints["tdnn4_bn"] = features
        endpoints["tdnn4_relu"] = features

        # Layer 5: [b, l, 1500]
        features = tf.layers.dense(features,
                                   1500,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                   name="tdnn5_dense")
        endpoints["tdnn5_dense"] = features
        features = relu(features, name='tdnn5_relu')
        features = tf.layers.batch_normalization(features,
                                                 momentum=0.99,
                                                 training=is_training,
                                                 name="tdnn5_bn")
        endpoints["tdnn5_bn"] = features
        endpoints["tdnn5_relu"] = features

        # Pooling layer
        # Layer 5: [b, l, 1500] --> [b, 3000]
        features = general_pooling(features, aux_features, endpoints, None, is_training)

        # Utterance-level network
        # Layer 6: [b, 512]
        features = tf.layers.dense(features,
                                   512,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                   name='tdnn6_dense')
        endpoints['tdnn6_dense'] = features
        features = relu(features, name='tdnn6_relu')
        features = tf.layers.batch_normalization(features,
                                                 momentum=0.99,
                                                 training=is_training,
                                                 name="tdnn6_bn")
        endpoints["tdnn6_bn"] = features
        endpoints["tdnn6_relu"] = features

        features = tf.layers.dense(features,
                                   512,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                   name='tdnn7_dense')
        endpoints['tdnn7_dense'] = features

        features = relu(features, name='tdnn7_relu')
        features = tf.layers.batch_normalization(features,
                                                     momentum=0.99,
                                                     training=is_training,
                                                     name="tdnn7_bn")
        endpoints["tdnn7_bn"] = features

        features = tf.layers.dense(features,
                                   output_target,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                   name='output_dense')
        endpoints['output_dense'] = features

    return features, endpoints


def lstm(features, is_training=None):
    tf.logging.info("Build a standard TDNN network.")
    relu = tf.nn.relu

    endpoints = collections.OrderedDict()
    with tf.variable_scope("tdnn", reuse=tf.AUTO_REUSE):
        # the orignal feature is [batch_size, length, dim]

        # Layer 1: [-2,-1,0,1,2] --> [b, l-4, 512]
        # conv2d + batchnorm + relu
        #features = tf.placeholder(tf.float32, shape=(64, None, 20))
        features = tf.layers.conv1d(features,
                                512,
                                5,
                                activation=None,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                name='tdnn1_conv')
        endpoints["tdnn1_conv"] = features
        features = relu(features, name='tdnn1_relu')
        features = tf.layers.batch_normalization(features,
                                                 momentum=0.99,
                                                 training=is_training,
                                                 name="tdnn1_bn")
        endpoints["tdnn1_bn"] = features
        endpoints["tdnn1_relu"] = features

        # Layer 2: [-2,  0,  2] --> [b , l-4, 512]
        # conv2d + batchnorm + relu
        # This is slightly different with Kaldi which use dilation convolution
        features = tf.layers.conv1d(features,
                                    512,
                                    5,
                                  #  dilation_rate=2,
                                    activation=None,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                    name='tdnn2_conv')
        endpoints["tdnn2_conv"] = features
        features = relu(features, name='tdnn2_relu')
        features = tf.layers.batch_normalization(features,
                                                 momentum=0.99,
                                                 training=is_training,
                                                 name="tdnn2_bn")
        endpoints["tdnn2_bn"] = features
        endpoints["tdnn2_relu"] = features

        # Layer 3: [-3, 0, 3] --> [b,  l-6, 512]
        # conv2d + batchnorm + relu
        # Still, use a non-dilation one
        features = tf.layers.conv1d(features,
                                    512,
                                    7,
                                  #  dilation_rate=3,
                                    activation=None,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                    name='tdnn3_conv')
        endpoints["tdnn3_conv"] = features
        features = relu(features, name='tdnn3_relu')
        features = tf.layers.batch_normalization(features,
                                                 momentum=0.99,
                                                 training=is_training,
                                                 name="tdnn3_bn")
        endpoints["tdnn3_bn"] = features
        endpoints["tdnn3_relu"] = features

        # Layer 4: [b, l, 512] --> [b, l, 512]
        features = tf.layers.dense(features,
                                   512,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                   name="tdnn4_dense")
        endpoints["tdnn4_dense"] = features
        features = relu(features, name='tdnn4_relu')
        features = tf.layers.batch_normalization(features,
                                                 momentum=0.99,
                                                 training=is_training,
                                                 name="tdnn4_bn")
        endpoints["tdnn4_bn"] = features
        endpoints["tdnn4_relu"] = features

        # Layer 5: [b, l, 1500]
        features = tf.layers.dense(features,
                                   1500,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                   name="tdnn5_dense")
        endpoints["tdnn5_dense"] = features
        features = relu(features, name='tdnn5_relu')
        features = tf.layers.batch_normalization(features,
                                                 momentum=0.99,
                                                 training=is_training,
                                                 name="tdnn5_bn")
        endpoints["tdnn5_bn"] = features
        endpoints["tdnn5_relu"] = features

        # Pooling layer
        # Layer 5: [b, l, 1500] --> [b, 3000]
        #features = statistics_pooling(features)

        features = attention(features, attention_size=1000)
        # Utterance-level network
        # Layer 6: [b, 512]
        features = tf.layers.dense(features,
                                   512,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                   name='tdnn6_dense')
        endpoints['tdnn6_dense'] = features
        features = relu(features, name='tdnn6_relu')
        features = tf.layers.batch_normalization(features,
                                                 momentum=0.99,
                                                 training=is_training,
                                                 name="tdnn6_bn")
        endpoints["tdnn6_bn"] = features
        endpoints["tdnn6_relu"] = features

        # Layer 7: [b, x]
        features = tf.layers.dense(features,
                                   512,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                   name='tdnn7_dense')
        endpoints['tdnn7_dense'] = features

        features = relu(features, name='tdnn7_relu')
        features = tf.layers.batch_normalization(features,
                                                     momentum=0.99,
                                                     training=is_training,
                                                     name="tdnn7_bn")
        endpoints["tdnn7_bn"] = features

        features = tf.layers.dense(features,
                                   num_classes,
                                   activation=None,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                   name='output_dense')
        endpoints['output_dense'] = features

    return features, endpoints

def lstm(features, is_training=None):
    """ build a lstm network.
    """
    # input feature [batch_size, length, dim]
    # length is time step
    chunk_size=40
    timesteps=chunk_size
    endpoints = collections.OrderedDict()

    with tf.variable_scope("tdnn", reuse=tf.AUTO_REUSE):
        #lstm_cells =  [tf.contrib.rnn.GRUBlockCellV2(num_units=256) for i in range(3)]
        lstm_cells =  [tf.contrib.rnn.LSTMCell(num_units=256, num_proj=128) for i in range(3)]
        lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)
        outputs, states = tf.nn.dynamic_rnn(cell=lstm, inputs=features, dtype=tf.float32, time_major=False)
        # the outputs is [batch_size, max_time, output_size]
        #output = tf.unstack(outputs, timesteps, 1) # the max_times [batch_size, output_size]
        #----average layer-------------
        #output = tf.reduce_mean(outputs, 1)
        #------------------------------
        # add attention layer
        #features = tf.layers.dense(output[-1], 14000,
        output = attention(outputs, attention_size=50)
        #------------------------------
        features = tf.layers.dense(output, 14000, 
                                  activation=None,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-2),
                                  name='output_dense')
    
        #features = tf.matmul(output[-1], weights['out']) + biases['out']
        endpoints['tdnn6_dense']=output
    return features, endpoints
