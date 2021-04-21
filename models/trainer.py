# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

import sys
import os
import random
import math

sys.path.insert(0, '../')

from nnet.graph import make_lstm_init_graph
from nnet.graph import make_lstm_graph
from nnet.graph import make_lstm_validation_graph
from data.pipeline import make_pipeline
from utils.gpu import average_models
from utils.thread import ThreadWithReturnValue
import numpy as np
import tensorflow as tf

#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

def compute_validation_loss(model, dev_egs_dir, gpu_id):
    device_id = "/gpu:" + str(gpu_id)
    sess_config = tf.ConfigProto(allow_soft_placement=True) 
    sess = tf.Session(config = sess_config)
    tfrecord_egs_file = "egs_tfrecords_{0}.scp".format(gpu_id)
    tfrecord_egs_file = os.path.join(dev_egs_dir, tfrecord_egs_file)
    assert os.path.exists(tfrecord_egs_file)
    tfrecord_egs_list = []
    with open(tfrecord_egs_file, 'r') as f:
        for line in f:
            line = line.strip()
            tfrecord_egs_list.append(line)

    with tf.device(device_id):
        pipeline = make_pipeline(sess, tfrecord_egs_list, batch_size=64, feature_dim=20)
        graph = make_lstm_validation_graph(pipeline)
        
    saver = tf.train.Saver()
    saver.restore(sess, model)
    step = 0
    total_loss = 0
    period_loss = 0
    average_loss = 0
    total_acc = 0
    period_acc = 0
    average_acc = 0
    accuracy = 0
    while True:
        try:
            accuracy, loss_val = sess.run([graph['accuracy'], graph['loss']])
            #loss_val = sess.run(graph['loss'])
            step += 1
            total_loss += loss_val
            period_loss += loss_val
            total_acc += accuracy
            period_acc += accuracy
            if step % 100 == 0:
                average_loss = period_loss / 100
                average_acc = period_acc / 100
                step_report = "the average val loss is: {2}, the step is from {0} to {1}, the acc is {3}".format(step-100, step, average_loss, average_acc)
                period_loss = 0
                period_acc = 0
                tf.logging.info(step_report)

        except tf.errors.OutOfRangeError:
            average_loss = total_loss / step
            average_acc = total_acc / step
            tf.logging.info("validation loss : {0}, accuracy: {1}".format(average_loss, average_acc))
            break

    sess.close()
    tf.logging.info("validation loss : {0}, accuracy: {1}".format(average_loss, average_acc))
    return (total_acc, total_loss, step)
 
def training_function_gpu(model_dir, best_model, egs_dir, gpu_id, iter, lrate, archive_idx, l2_regularize_factor, shuffle_seed):
    device_id = "/gpu:" + str(gpu_id)
    sess_config = tf.ConfigProto(allow_soft_placement=True) 
    sess = tf.Session(config = sess_config)
    tfrecord_egs_file = "egs_tfrecord_{0}.scp".format(archive_idx)
    #tfrecord_egs_file = "egs_tfrecord_1.scp"
    print(tfrecord_egs_file)
    tfrecord_egs_file = os.path.join(egs_dir, tfrecord_egs_file)
    assert os.path.exists(tfrecord_egs_file)
    tfrecord_egs_list = []
    with open(tfrecord_egs_file, 'r') as f:
        for line in f:
            line = line.strip()
            tfrecord_egs_list.append(line)

    with tf.device(device_id):
        pipeline = make_pipeline(sess, tfrecord_egs_list, batch_size=64, feature_dim=20, seed=shuffle_seed)
        graph = make_lstm_graph(pipeline, lrate, l2_regularize_factor)
    saver = tf.train.Saver() 
    last_model = "{0}/{1}.raw".format(model_dir, iter-1)
    last_model = best_model
    saver.restore(sess, last_model)
    #init = tf.global_variables_initializer()
    #sess.run(init) 
    #init = tf.local_variables_initializer()
    #sess.run(init)
    step = 0
    total_loss = 0
    period_loss = 0
    average_loss = 0
    while True:
        try:
            _, loss_val = sess.run([graph['train_op'], graph['loss_op']])
            step += 1
            total_loss += loss_val
            period_loss += loss_val
            if step % 100 == 0:
                average_loss = period_loss / 100
                step_report = "the average loss is: {2}, the step is from {0} to {1}, the gpu is {3}, lrate is {4}".format(step-100, step, average_loss, gpu_id, lrate)
                period_loss = 0
                print(step_report)
                tf.logging.info(step_report)
                

        except tf.errors.OutOfRangeError:
            model_name = model_dir + '/' + str(iter) + "." + str(gpu_id) + ".raw"
            save_path = saver.save(sess, model_name)
            average_loss = total_loss / step
            print("the average loss is : ", average_loss, gpu_id, lrate)
            print("Model save in: ", save_path)
            break

    sess.close()
    return average_loss

def compute_validation_loss_main(model, dev_egs_dir):
    threads = []
    num_jobs = 8
    for job in range(num_jobs):
        thread = ThreadWithReturnValue(target=compute_validation_loss, args=(model, dev_egs_dir, job))
        thread.start()
        threads.append(thread)

    loss_sum = 0
    acc_sum = 0
    total_step = 0
    for thread in threads:
        result = thread.join()
        acc_sum += result[0]
        loss_sum += result[1]
        total_step += result[2]
    return loss_sum / total_step, acc_sum / total_step

def init_model(model_dir, seed_model=None):
    sess_config = tf.ConfigProto(allow_soft_placement=True) 
    sess = tf.Session(config = sess_config)
    graph = make_lstm_init_graph(64,40,20)
    saver = tf.train.Saver() 
    model_name = "{0}/{1}.raw".format(model_dir, 0)
    init = tf.global_variables_initializer()
    sess.run(init) 
    init = tf.local_variables_initializer()
    sess.run(init)
    save_path = saver.save(sess, model_name)

def train(num_jobs_final, 
          num_jobs_initial,
					num_archives, 
					num_epochs, 
					model_dir, 
          num_jobs_step, 
          srand, 
					initial_effective_lrate, 
					final_effective_lrate, 
					train_egs_dir, 
					dev_egs_dir):
    best_model_idx = 0
    if os.path.exists(os.path.join(model_dir, "checkpoint")):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=model_dir)
        ckpt_list = ckpt.all_model_checkpoint_paths
        for model in ckpt_list:
            best_model_idx = int(model.split('/')[-1].split('.')[0])
    else:
        init_model(model_dir)
        best_model_idx = 0
    best_model = "{0}/{1}.raw".format(model_dir, best_model_idx)
 
    num_archives_to_process = int(num_epochs*num_archives)
    num_archives_processed = 0
    num_iters = int((num_archives_to_process*2) / (num_jobs_initial + num_jobs_final))
    best_dev_loss = sys.float_info.max 
    best_dev_acc = sys.float_info.min
    log_file_path=os.path.join(model_dir, 'log')


    for iter in range(num_iters):
        current_num_jobs = get_current_num_jobs(iter, num_iters, num_jobs_initial, 
                                                num_jobs_step, num_jobs_final)
        percent = num_archives_processed * 100.0 / num_archives_to_process 
        epoch = (num_archives_processed * num_epochs) / num_archives_to_process
        print("the percent is: ", percent, " in epoch: ", epoch)
        print("current_num_jobs: ", current_num_jobs)

        threads = []
        for job in range(1, current_num_jobs + 1):
            k = num_archives_processed + job - 1
            lrate = get_learning_rate(iter, current_num_jobs,
                                      num_iters,
                                      num_archives_processed,
                                      num_archives_to_process,
                                      initial_effective_lrate,
                                      final_effective_lrate)
            archive_idx = (k % num_archives) + 1
            l2_regularize_factor = 1.0 / current_num_jobs
            thread = ThreadWithReturnValue(target=training_function_gpu, args=(model_dir, best_model, train_egs_dir, job - 1, iter+1, lrate, archive_idx,
                                                                          l2_regularize_factor, srand+iter))

            thread.start()
            threads.append(thread)

        loss_list = []
        for thread in threads:
            k = thread.join()
            loss_list.append(k)
    
        # choose accepted models
        models_to_average = get_successful_models(loss_list)
        # combine model

        nnets_list = []
        for n in models_to_average:
            nnets_list.append("{0}/{1}.{2}.raw".format(model_dir, iter + 1, n))
    
        output_model = "{0}/{1}.raw".format(model_dir, iter+1)
        del_model = True
        average_models(nnets_list, output_model, del_model)
        average_training_loss = sum(loss_list) / len(loss_list)
        valid_loss, valid_acc = compute_validation_loss_main(output_model, dev_egs_dir)
        train_log_info = "iter {0} the average training loss is {1}, the lr is {2}".format(iter, average_training_loss, lrate)
        dev_log_info = "iter {0} the valid loss and acc is: {1}, {2}, the lr is: {3}".format(iter, valid_loss, valid_acc, lrate)
        tf.logging.info(train_log_info)
        tf.logging.info(dev_log_info)
        num_archives_processed += current_num_jobs
        if (best_dev_loss > valid_loss) :
            log_file = open(log_file_path, 'r+')
            best_dev_loss =  valid_loss
            best_dev_acc = valid_acc
            dev_log_info = "best dev loss is: {0}, best acc is: {1}\n".format(best_dev_loss, best_dev_acc)
            best_model_info = "best model is: {0}\n".format(output_model)
            log_file.write(dev_log_info)
            log_file.write(best_model_info)
            best_model = output_model
            tf.logging.info(dev_log_info)
            tf.logging.info(best_model_info)
            log_file.close()
    tf.logging.info("training is finished!!!")

def test_train():
    num_jobs_final = 1
    num_jobs_initial = 8
    num_archives = 205
    num_epochs = 20
    num_archives_to_process = int(num_epochs*num_archives)
    num_archives_processed = 0
    num_iters = int((num_archives_to_process*2) / (num_jobs_initial + num_jobs_final))
    num_jobs_step=1
    srand=1000

    initial_effective_lrate=0.001
    final_effective_lrate=0.0001
    model_dir="./exp"

    training_function_gpu(model_dir, 0, 1, 0.001, 1,    
                          0.5, srand)

def main(argv):
    test_train()
    exit()
    train()

if __name__ == '__main__':
    main(sys.argv)
