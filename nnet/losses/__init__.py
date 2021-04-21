# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

__all__ = [ 
        "LOSSES" 
]

import tensorflow as tf

LOSSES = {
  "softmax_cross_entropy":
      lambda  labels, input, config: tf.losses.softmax_cross_entropy(onehot_labels, logits, **config),
  "sigmoid_cross_entropy": tf.losses.sigmoid_cross_entropy(multi_class_labels, logits, **config),
  "mean_pairwise_squared_error"  : tf.losses.mean_pairwise_squared_error(labels, predictions, **config),
  "sparse_softmax_cross_entropy": tf.losses.sparse_softmax_cross_entropy(labels, logits, **config),
  "mean_squared_error": tf.losses.mean_squared_error(labels, predictions, **config),
}


