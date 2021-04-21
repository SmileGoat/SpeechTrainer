# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)


__all__ = [ 
        "METRICS" 
]

import tensorflow as tf

METRICS = {
    "accuracy": lambda labels, predictions, config: tf.metrics.accuracy(labels, predictions, **config),
    "acu": lambda labels, predictions, config: tf.metrics.auc(labels, predictions, **config),
    "precision": lambda labels, predictions, config: tf.metrics.precision(labels, predictions, **config),
}


