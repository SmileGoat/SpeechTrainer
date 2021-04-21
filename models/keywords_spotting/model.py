# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

def _weights_add(weights1, weights2):
    weights = [w1 + w2 for w1, w2 in zip(weights1, weights2)]
    return weights

def _weights_div(weights, num):
    weights = [w / num for w in weights]
    return weights

def _weights_time(weights, num):
    weights = [w * num for w in weights]
    return weights

def average_model(model, model_name, ckpt_list):
    avg_weights = None
    for idx, ckpt in enumerate(ckpt_list):
        model.load_weights(ckpt)
        model_weights = model.get_weights()
        if idx == 0:
            avg_weights = model_weights
            continue
        avg_weights = _weights_add(avg_weights, model_weights)
    avg_weights = _weights_div(avg_weights, len(ckpt_list))
    model.set_weights(avg_weights)
    model.save_weights(model_name)

