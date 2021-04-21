# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)


import sys
import os
import random

sys.path.insert(0, 'utils')

def Usage():
    print("Usage: {0} percent train_dev train dev".format(sys.argv[0]))
    print("the percent means dev/train_dev = percent")


def main(percent, train_dev, train, dev):
    spk2utt_fd = os.path.join(train_dev, "spk2utt")
    assert os.path.exists(spk2utt_fd)

    train_spk2utt = os.path.join(train, "spk2utt")
    dev_spk2utt = os.path.join(dev, "spk2utt")
    train_spk2utt_writer = open(train_spk2utt, 'w')
    dev_spk2utt_writer = open(dev_spk2utt, 'w')
    train_feats_writer = open(os.path.join(train, 'feats.scp'), 'w')
    dev_feats_writer = open(os.path.join(dev, 'feats.scp'), 'w')

    train_dev_feats_dict = dict()
    train_dev_fd = os.path.join(train_dev, 'feats.scp')
    with open(train_dev_fd, 'r') as f:
        for line in f:
            utt_id, feat = line.strip().split()
            train_dev_feats_dict[utt_id] = feat
    
    with open(spk2utt_fd, 'r') as f:
        for line in f:
            line_arr = line.strip().split()
            spk = line_arr[0]
            utts = line_arr[1:]
            dev_num = int(len(utts)  * percent)
            random.shuffle(utts)
            dev_line = spk + " " + " ".join(utts[0:dev_num]) + '\n'
            train_line = spk + " " + " ".join(utts[dev_num:]) + '\n'
            train_spk2utt_writer.write(train_line)
            dev_spk2utt_writer.write(dev_line)
            for utt in utts[0:dev_num] :
                feat = train_dev_feats_dict.get(utt)
                if feat != None:
                    dev_feat_line = utt + " " + feat + "\n"
                    dev_feats_writer.write(dev_feat_line)
                else:
                    print(utt , " not exist in train_dev/feats.scp")

            for utt in utts[dev_num:] :
                feat = train_dev_feats_dict.get(utt)
                if feat != None:
                    train_feat_line = utt + " " + feat + "\n"
                    train_feats_writer.write(train_feat_line)
                else:
                    print(utt , " not exist in train_dev/feats.scp")

if __name__ == '__main__':
    if len(sys.argv) != 5 :
        Usage()
    percent = float(sys.argv[1])
    train_dev_dir = sys.argv[2]
    train_dir = sys.argv[3]
    dev_dir = sys.argv[4]
    main(percent, train_dev_dir, train_dir, dev_dir)


