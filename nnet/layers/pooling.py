# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

VAR2STD_EPSILON = 1e-12

def statistics_pooling(features):
    """Statistics pooling
        features:  [batch, length, dim].
        result [mean, stddev] with shape [batch, dim].
    """
    with tf.variable_scope("stat_pooling"):
        mean = tf.reduce_mean(features, axis=1, keepdims=True, name="mean")
        variance = tf.reduce_mean(tf.squared_difference(features, mean), axis=1, keepdims=True, name="variance")
        # mean is [batch, dim]
        # variance is [batch, dim]
        mean = tf.squeeze(mean, 1) 
        variance = tf.squeeze(variance, 1) 

        mask = tf.to_float(tf.less_equal(variance, VAR2STD_EPSILON))
        variance = (1.0 - mask) * variance + mask * VAR2STD_EPSILON
        stddev = tf.sqrt(variance)

        stat_pooling = tf.concat([mean, stddev], 1, name="concat")

    return stat_pooling

