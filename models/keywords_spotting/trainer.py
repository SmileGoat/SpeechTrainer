# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

import tensorflow as tf
import os
import sys
import time

def cross_entropy_loss(x, y):
    y = tf.cast(y, tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    return tf.reduce_mean(loss)

def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def test_trainer(tfrecords_list, model_dir, iter, lrate):
    num_epoch = 10000
    training_step = 100000
    device_id = "/gpu:0"
    train_dataset = read_dataset_from_tfrecords(tfrecords_list)
    train_dataset = train_dataset.repeat(num_epoch).shuffle(1024).batch(128).prefetch(1)
    last_model = "{0}/{1}.raw".format(model_dir, iter-1)

    with tf.device('/cpu:0'):
        neural_net = DNN(640)

    neural_net.load_weights(last_model)
    optimizer = tf.optimizers.Adam(lrate)
    trainable_variables = neural_net.trainable_variables
    display_step = 100

    ts = time.time()
    for step, feature in enumerate(train_dataset.take(training_step), 1):
        with tf.device(device_id):
            with tf.GradientTape() as g:
                y_pred = neural_net(feature['nnet_input'], is_training=True)
                loss = cross_entropy_loss(y_pred, feature['nnet_output'])
                gradient = g.gradient(loss, trainable_variables)
        with tf.device('/cpu:0'):
            optimizer.apply_gradients(zip(gradient, trainable_variables))
        if step % display_step == 0 or step == 1:
            dt = time.time() - ts
            y_pred = neural_net(feature['nnet_input'], is_training=False)
            loss = cross_entropy_loss(y_pred, feature['nnet_output'])
            acc = accuracy(y_pred, feature['nnet_output'])
            step_log = "step: {0}, loss: {1}, acc:{2}, time:{3} ".format(step, loss, acc, dt)
            print(step_log)
            ts = time.time()
        if step % 10000 == 0:
            model_id = step / 10000
            model_dir='~/workspace/tensorflow_speech_trainer/models/keywords_spotting/model_store'
            model_name = str(model_id) + ".raw"
            new_model_name = os.path.join(model_dir, model_name)
            neural_net.save_weights(new_model_name)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords_list', type=str, help='tfrecords list')
    args = parser.parse_args()
    tfrecords_list_file = args.tfrecords_list
    tfrecords_list = []
    with open(tfrecords_list_file, 'r') as f:
        for line in f:
            line = line.strip()
            line = line[line.find(' ') + 1:].lstrip()
            tfrecords_list.append(line)

    model_dir='~/workspace/tensorflow_speech_trainer/models/keywords_spotting/model_store'
    neural_net = DNN(640)
    model_name = '0.raw'
    model_init = os.path.join(model_dir, model_name)
    neural_net.save_weights(model_init)
    iter = 1
    lrate = 0.01
    model_list = []
    tmp = []
    tmp.append(2.0)
    tmp.append(3.0)
    tmp.append(4.0)
    for ckpt in tmp:
        model_name = str(ckpt) + '.raw'
        model_list.append(os.path.join(model_dir, model_name)) 
    final_model = os.path.join(model_dir, 'final.raw')
    average_model(neural_net, final_model, model_list)

