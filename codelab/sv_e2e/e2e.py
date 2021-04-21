# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

import tensorflow as tf

def normalize(x):
    result = x / tf.sqrt(tf.reduce_sum(x**2, axis=-1, keep_dims=True) + 1e-6)
    return result

def cossin(x, y, axis=None, keep_dims=False):
    x_l2 = tf.sqrt(tf.reduce_sum(x**2, axis=-1, keepdims=True) + 1e-6)
    y_l2 = tf.sqrt(tf.reduce_sum(x**2, axis=-1, keepdims=True) + 1e-6)
    return tf.reduce_sum(x*y, axis = axis, keep_dims=keep_dims)/(x_l2*y_l2)

def similarity(embedding, spk_num, utt_num, w, b):
    #input is [N, M, dim]
    N = spk_num
    M = utt_num
    center = tf.reduce_mean(embedding, axis=1)
    center_remove_self = tf.reduce_sum(embedding, axis=1, keepdims=True) - embedding / (M - 1)
		# S matrix is [N*M, N] N spks
		S = []
		for j in range(N):
				similarity = []
				for k in range(N):
					  if k == j:
								cos_distant = cossin(center_remove_self[k, :, :], embedding[j, :, :], axis=1, keepdims=True) # [1, dim] , [M, dim] --> [M, 1] (eq9, k=j)
						else:
								cos_distant = cossin(center[k:k+1, :], embedding[j, :, :], axis=1, keepdims=True) # [1,dim], [M, dim] -->[M, 1] (eq9,others)
						similarity.append(cos_distant)
				S.append(tf.concat(similarity, axis=1)) # S is N * [M, N]
		S = tf.concat([S], axis=0) # final s is [N, M, N]
	  return S*w + b

