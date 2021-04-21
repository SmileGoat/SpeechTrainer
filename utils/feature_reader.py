# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

import utils.kaldi_compressed_matrix as cmatrix_reader
import random
import os

class SpeakerFeatureReader(object):
    """Read kaldi features"""
    def __init__(self, data):
        """
        Args:
            data: The kaldi data directory.
        """
        self.fd = {}
        self.data = data
        self.matrix_reader = cmatrix_reader.CompressMatrix()
        self.dim = self.get_dim()
        self.utt2num_frames = {}
        # Load utt2num_frames that the object can find the length of the utterance quickly.
        assert os.path.exists(os.path.join(data, "utt2num_frames")), "[Error] Expect utt2num_frames exists in %s " % data
        with open(os.path.join(data, "utt2num_frames"), 'r') as f:
            for line in f.readlines():
                utt, length = line.strip().split(" ")
                self.utt2num_frames[utt] = int(length)

    def get_dim(self):
        with open(os.path.join(self.data, "feats.scp"), "r") as f:
            dim = self.read(f.readline().strip())[0].shape[1]
        return dim

    def get_mean(self):
        with open(os.path.join(self.data, "feats.scp"), "r") as f:
            dim = self.read(f.readline().strip())[0].shape[1]
        return dim

    def close(self):
        for name in self.fd:
            self.fd[name].close()

    def read(self, file_name, length=None, shuffle=False, start=None):
        utt, file_or_fd = file_or_fd.split(" ")
        (filename, offset) = file_or_fd.rsplit(":", 1)
        if filename not in self.fd:
            fd = open(filename, 'rb')
            assert fd is not None
            self.fd[filename] = fd
        # Move to the target position
        #self.fd[filename].seek(int(offset))
        #try:
        self.matrix_reader.read(self.fd[filename], int(offset))
        mat = self.matrix_reader.numpy()
        #except:
        #    raise IOError("Cannot read features from %s" % file_or_fd)

        if length is not None:
            if start is None:
                num_features = mat.shape[0]
                length = num_features if length > num_features else length
                start = random.randint(0, num_features - length) if shuffle else 0
                mat = mat[start:start + length, :]
            else:
                assert not shuffle, "The start point is specified, thus shuffling is invalid."
                mat = mat[start:start + length, :]
        return mat, start
-
