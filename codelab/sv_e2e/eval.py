# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

import numpy as np
from utils.kaldi_io import read_vec_flt_scp

class Eval(object):
    def __init__(self, utt2spk, enroll_scp, eval_scp, score_file, w, b):
        super().__init__()
        self.w = w
        self.b = b
        self.score_file = score_file
        self.enroll_spks = []
        enroll_center = []
        # attention: the enroll center is: spk mean_feats
        for spk, vec in read_vec_flt_scp(enroll_scp):
           self.enroll_spks.append(spk)
           enroll_center.append(vec)
        self.enroll_centers = np.array(enroll_center)

    def normalize(self,x):
        result = x / np.sqrt(np.sum(np.power(x,2), axis=-1, keepdims=True) + 1e-6)
        return result

    def cossin(self, x, y):
        x_norm = normalize(x)
        y_norm = normalize(y)
        return np.sum(x_norm*y_norm, axis=1, keepdims=True)

    def similarity(self, embedding, center, w, b):
        # embedding : 1 * dim 
        # center : N(num_enroll) * dim
        cos_distant = cossin(embedding, center) # [N, 1]
        return cos_distant*abs(w) + b

    def match_enroll_eval_spk(self, eval_spk):
        match_result = []
        for enroll_spk in self.enroll_spks:
            match = ""
            if enroll_spk == eval_spk:
                match = "target"
            else:
                match = "nontarget"
            match_result.append(match)
        return match_result

    def eval_result(self):
        # socre_line should be : spk utt score target
        score_writer = open(self.score_file, 'w')
        for utt, vec in read_vec_flt_scp(self.eval_scp):
            eval_spk = self.utt2spk(utt)
            targets = match_enroll_eval_spk(eval_spk)
            eval_score = similarity(vec, self.enroll_centers, self.w, self.b)
            assert len(targets) == eval_score.shape[0], "target and similarity are dismatch"
            for idx in range(len(targets)):
                score_str = self.enroll_spks[idx] + " " + utt + " " + str(eval_score[idx]) + " " + targets[idx]
                score_writer.write(score_str)
        score_writer.close()

'''
# test the function
def normalize(x):
    result = x / np.sqrt(np.sum(np.power(x,2), axis=-1, keepdims=True) + 1e-6)
    return result

def cossin(x, y):
    x_norm = normalize(x)
    y_norm = normalize(y)
    return np.sum(x_norm*y_norm, axis=1, keepdims=True)

def similarity(embedding, center, w, b):
    # embedding : 1 * dim 
    # center : N(num_enroll) * dim
    cos_distant = cossin(embedding, center) # [N, 1]
    return cos_distant*abs(w) + b

if __name__ == "__main__":
    w = 1.0
    b = 0
    enroll_center = np.array([[0,1,0], [8.1,0,1], [0,1.4,0], [22,1,0], [1,0,0], [9.1,0,10.9]])
    eval_embedd1 = np.array([2.04,1,2.34])
    eval_embedd2 = np.array([0.35,1,0.28])
    print(similarity(eval_embedd1, enroll_center, w, b))
    print(similarity(eval_embedd2, enroll_center, w, b))
    # the result is  
    [[0.30660985]
     [0.70868034]
     [0.30660992]
     [0.63876168]
     [0.62548409]
     [0.95161781]]
    [[0.91252796]
     [0.34828497]
     [0.91252818]
     [0.36049129]
     [0.31938479]
     [0.40082549]]
'''
