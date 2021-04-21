# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

from tensorflow.keras import Model, layers

class DNN(Model):
    """ build a dnn based keyword spotting system
    implementation of the paper below: 
    Chen G, Parada C, Heigold G. Small-footprint keyword spotting using deep neural networks
    """
    def __init__(self, feature_dim, hidden_units=256, num_classes=5):
        super(DNN, self).__init__()
        self.layer1 = layers.Dense(feature_dim)
        self.layer2 = layers.Dense(hidden_units)
        self.layer3 = layers.Dense(hidden_units)
        self.layer4 = layers.Dense(hidden_units)
        self.out = layers.Dense(num_classes)

    def call(self, feature, is_training=False):
        # feature is [batch, (left_context + 1 + right_context) * dim] 
        x = self.layer1(feature)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.out(x)
        if not is_training:
          x = tf.nn.softmax(x)
        return x


