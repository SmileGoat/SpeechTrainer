# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

import tfrecords_util as tf_util
import random

def get_feature_and_idx(scp_line):
    line_list = scp_line.rstrip().split()
    wav_id = line_list[0]
    file_info_list = line_list[1].split(':')
    file_id = file_info_list[0]
    file_pos = int(file_info_list[1])
    file_obj = open(file_id, 'rb')
    return wav_id, file_obj, file_pos

class FeatureReader(object):
    def __init__(self, data_dir):
        self.fd = {}
        self.utt2num_frames = {}
        self.feats_scp = os.path.join(data_dir, "feats.scp")
        self.dim = self.get_dim()
        self.utt2spk = {}
        self.spk2int = {}
        self.wav2feat = {}

        with open(os.path.join(data_dir, "utt2num_frames"), 'r') as f:
            for line in f.readlines():
                utt, length = line.strip().split(" ")
                self.utt2num_frames[utt] = int(length)

        with open(os.path.join(data_dir, "utt2spk"), 'r') as f:
            for line in f.readlines():
                utt, spk = line.strip().split(" ")
                self.utt2spk[utt] = spk

        with open(os.path.join(data_dir, "spk2int"), 'r') as f:
            for line in f.readlines():
                spk, spk_id = line.strip().split(" ")
                self.utt2int[spk] = int(spk_id) 

        with open(os.path.join(), "feats.scp") as f:
            for line in f.readlines():
                wav, feat = line.strip().split(" ")
                self.feats[wav] = feat

    def get_dim(self):
        compress_matrix = cm.CompressMatrix()
        with open(self.feats_scp, "r") as f:
            for line in f.readlines():
                wav_id, file_obj, pos = get_feature_and_idx(line)
                compress_matrix.read(file_obj, pos)
                feat_data = compress_matrix.numpy()
                self.dim = feat_data.shape[1]
                break
        f.close()


def write_tfrecords(dir, wav_id, feat_data, spk2id):
    rows = feat_data.shape[0]
    cols = feat_data.shape[1]
    spk = wav_id.split('_')[0]
    spk_id = spk2id[spk]
    feature = {
        'shape' : tf_util._int64_feature(feat_data.shape),
        'nnet_input' : tf_util._float_feature(feat_data.flatten()),
        'nnet_output' : tf_util._float_feature(spk_ids),
    }
    tfrecord_name = wav_id.split('.')[0] + '.tfrecord'
    tf_feat_file = dir + '/' + tfrecord_name
    tfwriter = tf.io.TFRecordWriter(tf_feat_file)
    tfwriter.write(tf_util.make_example(feature))
    return tf_feat_file

def process_dict(spkidx_file):
    spk_dict = {}
    with open(spkidx_file, 'r') as id_file:
        for line in id_file:
           line_arr = line.split()
           spk_dict[line_arr[0]] = int(line_arr[1])
    return spk_dict

def shuffle_egs(buffer_size:int = None, seed=124, range_egs:list = None):
    if buffer_size is None:
        #shuffle all egs 
        random.shuffle(range_egs)
        return range_egs

    egs_shuffle = []
    egs = {}
    random.seed(seed) 
    for eg in range_egs:
        index:int = random.randint(0, buffer_size - 1)
        if not egs.has_key(index):
            egs[index] =  eg
        else :
            egs_shuffle.append(egs[index])
            egs[index] = eg
    return egs_shuffle 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
#        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--feature_input_scp', type=str, help='feature input scp')
    parser.add_argument('--range_dir', type=str, help='range')
    parser.add_argument('--spk2idx', type=str, help='spk2idx')
    parser.add_argument('--utt2spk', type=str, help='utt2spk')
    parser.add_argument('--tfrecords_scp', type=str, help='tfrecords scp')
    parser.add_argument('--tfrecords_dir', type=str, help='tfrecords dir')
    parser.add_argument('--batch_size', type=int, help='batch_size')

    args = parser.parse_args()
    feats_scp = args.feature_input_scp
    tfrecords_scp = args.tfrecords_scp
    tfrecords_dir = args.tfrecords_dir
    spk2idx_file = args.spk2id
    batch_size = args.batch_size
    spk2id = process_dict(spk2idx_file)
    compress_matrix = cm.CompressMatrix()
    tf_scp = open(tfrecords_scp, 'w')

    shuffle_egs(buffer_size = 1000, seed=124, range_egs)
    with open(feats_scp, 'r') as feats_file:
        batch_idx = 0
        batch_features = []
        spkid_list = []
        for line in feats_file:
            batch_idx += 1
            wav_id, file_obj, pos = get_feature_and_idx(line)
            compress_matrix.read(file_obj, pos)
            feat_data = compress_matrix.numpy()
            batch_features.append(feat_data)
            if spk2id.has_key(wav_id):
                spk_id = spk2id(wav_id)
                spkid_list.append(spk_id)
            else:
                logging.warning("no speaker has wav: " + wav_id + "\n")
                batch_idx -= 1
                continue

            if batch_idx == batch_size:
                tf_feat_file = write_tfrecords(tfrecords_dir, batch_features, spkid_list)
                tf_feat_file = speak + " " + tf_feat_file + "\n"
                spkid_list.clear()
                batch_features.clear()
                tf_scp.write(tf_feat_file)
    tf_scp.close()
