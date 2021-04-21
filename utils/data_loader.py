# the writer is LIUYI
# the file is from his repo

import os
import sys
import random
import numpy as np
import tensorflow as tf
import utils.feature_reader as feature_reader
from multiprocessing import Process, Queue, Event
import logging

logger = logging.getLogger(__name__)

def get_speaker_info(data):
    """Get speaker information from the data directory.
    Args:
        data: The kaldi data directory.
        spklist: The spklist file gives the index of each speaker.
    :return:
        spk2features: A dict. The key is the speaker id and the value is the segments belonging to this speaker.
        features2spk: A dict. The key is the segment and the value is the corresponding speaker id.
        spk2index: A dict from speaker NAME to speaker ID. This is useful to get the number of speakers. Because
                   sometimes, the speakers are not all included in the data directory (like in the valid set).

        segment format: "utt_name filename:offset"
    """
    assert (os.path.isdir(data))
    spk2index = {}
    with open(os.path.join(data, 'spk2int'), "r") as f:
        for line in f.readlines():
            spk, index = line.strip().split(" ")
            spk2index[spk] = int(index)

    utt2num_frames = {}
    with open(os.path.join(data, "utt2num_frames"), "r") as f:
        for line in f.readlines():
            utt, nums = line.strip().split(" ")
            utt2num_frames[utt] = int(nums)

    utt2spk = {}
    spk_min_frames = {}
    with open(os.path.join(data, "spk2utt"), "r") as f:
        for line in f.readlines():
            spk, utts = line.strip().split(" ", 1)
            min_frame_length = sys.maxsize
            for utt in utts.split(" "):
                utt2spk[utt] = spk2index[spk]
                if utt in utt2num_frames.keys():
                    frame_length = utt2num_frames[utt]
                min_frame_length = frame_length if frame_length < min_frame_length else min_frame_length 
            spk_min_frames[spk2index[spk]] = min_frame_length

    spk2features = {}
    features2spk = {}
    with open(os.path.join(data, "feats.scp"), "r") as f:
        for line in f.readlines():
            (key, rxfile) = line.strip().split(' ')
            spk = utt2spk[key]
            if spk not in spk2features:
                spk2features[spk] = []
            spk2features[spk].append(key + ' ' + rxfile)
            features2spk[key + ' ' + rxfile] = spk

    return spk2features, features2spk, spk2index, utt2num_frames, spk_min_frames

def get_feature_mean(speakers, spk2features, feature_reader):
    batch_speakers = random.sample(speakers, 100)
    feature_arr = []
    for speaker in batch_speakers:
        feature_list =  spk2features[speaker]
        for feat in feature_list:
            feat_data, _ = feature_reader.read(feat)
            feature_arr.append(feat_data)
    feature = np.concatenate(feature_arr) 
    mean = feature.mean(0)
    print(mean)
    return mean

class KaldiDataRandomReader(object):
    """Used to read data from a kaldi data directory."""

    def __init__(self, data_dir, num_parallel=1, num_speakers=None, num_segments=None, min_len=None, max_len=None, shuffle=True):
        """ Create a data_reader from a given directory.

        Args:
            data_dir: The kaldi data directory. 
                      must include feat.scp, spk2int, spk2utt, utt2num_frames
            spklist: The spklist tells the relation between speaker and index.
            num_parallel: The number of threads to read features.
            num_speakers: The number of speakers per batch.
            num_segments: The number of semgents per speaker.
              batch_size = num_speakers * num_segments
              When num_segments = 1, the batch is randomly chosen from n speakers,
              which is used for softmax-like loss function. While we can sample multiple segments for each speaker,
              which is used for triplet-loss or GE2E loss.
            min_len: The minimum length of the sampled sequence.
            max_len: The maximum length of the sampled sequence.
            shuffle: Load the feature from the 0-th frame or a random frame.
        """
        self.data = data_dir
        self.num_speakers = num_speakers
        self.num_segments = num_segments
        self.min_len = min_len
        self.max_len = max_len
        self.shuffle = shuffle

        # We process the data directory and fetch speaker information
        self.spk_reader = feature_reader.SpeakerFeatureReader(data_dir)
        self.dim = self.spk_reader.get_dim() 
        self.spk2features, self.features2spk, self.spk2index, self.utt2num_frames, self.spk_min_frames = get_speaker_info(data_dir)
        self.speakers = list(self.spk2features.keys())
       # self.mean = get_feature_mean(self.speakers, self.spk2features, self.spk_reader)
        self.num_total_speakers = len(list(self.spk2index.keys()))
        self.num_parallel_datasets = num_parallel
        if self.num_parallel_datasets != 1:
            raise NotImplementedError("When num_parallel_datasets != 1, we got some strange problem with the dataset. Waiting for fix.")

    def set_batch(self, num_speakers, num_segments):
        """Set the batch-related parameters

        Args:
            num_speakers: The number of speakers per batch.
            num_segments: The number of semgents per speaker.
        """
        self.num_speakers = num_speakers
        self.num_segments = num_segments

    def set_length(self, min_len, max_len):
        """Set the length of the sequence

        Args:
            min_len: The minimum length of the sampled sequence.
            max_len: The maximum length of the sampled sequence.
        """
        self.min_len = min_len
        self.max_len = max_len

    def get_min_length(self, features_list):
        min_length = sys.maxsize 
        for feat in features_list:
           feat_id = feat.strip().split()[0] 
           length = self.utt2num_frames(feat_id)
           min_length = length if length < min_length else min_length
        return min_length


    def batch_random(self):
        """Randomly load features to form a batch

        This function is used in the load routine to feed data to the dataset object
        It can also be used as a generator to get data directly.
        """
        feature_reader = self.spk_reader
        speakers = self.speakers
        #speakers = list(range(0, 100))
        if self.num_total_speakers < self.num_speakers:
            print(
                "[Warning] The number of available speakers are less than the required speaker. Some speakers will be duplicated.")
            speakers = self.speakers * (int(self.num_speakers / self.num_total_speakers) + 1)

        while True:
            #todo there is somthing wrong with batch_length, beacuse some feature length < batch_length
            batch_length = random.randint(self.min_len, self.max_len)
            batch_speakers = random.sample(speakers, self.num_speakers)
            labels = np.zeros((self.num_speakers * self.num_segments), dtype=np.int32)
            min_batch_length = sys.maxsize
            for speaker in batch_speakers:
                min_frames = self.spk_min_frames[speaker]
                min_batch_length = min_frames if min_frames < min_batch_length else min_batch_length
            batch_length = min_batch_length

            features_arr = []
            for i, speaker in enumerate(batch_speakers):
                labels[i * self.num_segments:(i + 1) * self.num_segments] = speaker
                feature_list = self.spk2features[speaker]
                if len(feature_list) < self.num_segments:
                    feature_list *= (int(self.num_segments / len(feature_list)) + 1)
                # Now the length of the list must be greater than the sample size.
                speaker_features = random.sample(feature_list, self.num_segments)
                for j, feat in enumerate(speaker_features):
                    wav_name = feat.split(" ")[0] 
                    feat_data, _ = feature_reader.read(feat, batch_length, shuffle=self.shuffle)
                    features_arr.append(feat_data)
            features = np.concatenate([features_arr])
            yield (features, labels)

    def load_dataset(self):
        """ Load data from Kaldi features and return tf.dataset.
        The function is useful for training, since it randomly loads features and labels from N speakers,
          with K segments per speaker.
        The batch is sampled randomly, so there is no need to do shuffle again.

        :return: A nested tensor (features, labels)
        """
        batch_size = self.num_speakers * self.num_segments
        if self.num_parallel_datasets == 1:
            # Single thread loading
            dataset = tf.data.Dataset.from_generator(self.batch_random, (tf.float32, tf.int32),
                                                     (tf.TensorShape([batch_size, None, self.dim]),
                                                      tf.TensorShape([batch_size])))
        else:
            # Multiple threads loading
            # It is very strange that the following code doesn't work properly.
            # I guess the reason may be the py_func influence the performance of parallel_interleave.
            dataset = tf.data.Dataset.range(self.num_parallel_datasets).apply(
                tf.contrib.data.parallel_interleave(
                    lambda x: tf.data.Dataset.from_generator(self.batch_random, (tf.float32, tf.int32),
                                                             (tf.TensorShape([batch_size, None, self.dim]),
                                                              tf.TensorShape([batch_size]))),
                    cycle_length=self.num_parallel_datasets,
                    sloppy=False))
        dataset = dataset.prefetch(1)
        return dataset

class KaldiDataRandomQueue(object):
    """A queue to read features from Kaldi data directory."""

    def __init__(self, data_dir, num_parallel=1, max_qsize=20, num_speakers=None, num_segments=None, min_len=None, max_len=None, shuffle=True, sample_with_prob=False):
        """ Create a queue from a given directory.

        This is basically similar with KaldiDataRead. The difference is that KaldiDataReader uses tf.data to load
        features and KaldiDataQueue uses multiprocessing to load features which seems to be a better choice since
        the multiprocessing significantly speed up the loading in my case. If you can make parallel_interleave works,
        it is definitely more convenient to use KaldiDataReader because it's more simple.

        Args:
            data_dir: The kaldi data directory.
            spklist: The spklist tells the mapping from the speaker name to the speaker id.
            num_parallel: The number of threads to read features.
            max_qsize: The capacity of the queue
            num_speakers: The number of speakers per batch.
            num_segments: The number of semgents per speaker.
              batch_size = num_speakers * num_segments
              When num_segments = 1, the batch is randomly chosen from n speakers,
              which is used for softmax-like loss function. While we can sample multiple segments for each speaker,
              which is used for triplet-loss or GE2E loss.
            min_len: The minimum length of the sampled sequence.
            max_len: The maximum length of the sampled sequence.
            shuffle: Loading data from the 0-th frame or a random frame.
            sample_with_prob: Sample the speaker and utt with the probability according to the data length.
        """
        self.data = data_dir
        self.num_speakers = num_speakers
        self.num_segments = num_segments
        self.min_len = min_len
        self.max_len = max_len
        self.num_parallel_datasets = num_parallel
        self.shuffle = shuffle
        self.sample_with_prob = sample_with_prob

        if self.sample_with_prob:
            logger.info("The training examples are sampled with probability.")

        # We process the data directory and fetch speaker information.
        self.spk2features, self.features2spk, self.spk2index, self.utt2num_frames, self.spk_min_frames = get_speaker_info(data_dir)
        # We also load #frames for each speaker and #frames for each utt
        self.utt2num_frames = {}
        with open(os.path.join(data_dir, "utt2num_frames"), 'r') as f:
            for line in f.readlines():
                utt, n = line.strip().split(" ")
                self.utt2num_frames[utt] = int(n)

        self.spk2num_frames = {}
        for spk in self.spk2features:
            n = 0
            for utt in self.spk2features[spk]:
                n += self.utt2num_frames[utt.split(" ")[0]]
            self.spk2num_frames[spk] = n

        # The number of speakers should be
        self.num_total_speakers = len(list(self.spk2index.keys()))

        # The Queue is thread-safe and used to save the features.
        self.queue = Queue(max_qsize)
        self.stop_event = Event()

        # And the prcesses are saved
        self.processes = []

    def set_batch(self, num_speakers, num_segments):
        """Set the batch-related parameters

        Args:
            num_speakers: The number of speakers per batch.
            num_segments: The number of semgents per speaker.
        """
        self.num_speakers = num_speakers
        self.num_segments = num_segments

    def set_length(self, min_len, max_len):
        """Set the length of the sequence

        Args:
            min_len: The minimum length of the sampled sequence.
            max_len: The maximum length of the sampled sequence.
        """
        self.min_len = min_len
        self.max_len = max_len

    def start(self):
        """Start processes to load features
        """
        self.processes = [Process(target=batch_random, args=(self.stop_event,
                                                             self.queue,
                                                             self.data,
                                                             self.spk2features,
                                                             self.spk2num_frames,
                                                             self.utt2num_frames,
                                                             self.num_total_speakers,
                                                             self.num_speakers,
                                                             self.num_segments,
                                                             self.min_len,
                                                             self.max_len,
                                                             self.shuffle,
                                                             i,
                                                             self.sample_with_prob))
                          for i in range(self.num_parallel_datasets)]
        for process in self.processes:
            process.daemon = True
            process.start()

    def fetch(self):
        """Fetch data from the queue"""
        return self.queue.get()

    def stop(self):
        """Stop the threads

        After stop, the processes are terminated and the queue may become unavailable.
        """
        self.stop_event.set()
        print("Clean the data queue that subprocesses can detect the stop event...")
        while not self.queue.empty():
            # Clear the queue content before join the threads. They may wait for putting the data to the queue.
            self.queue.get()
        time.sleep(3)
        for process in self.processes:
            # TODO: fix the join problem
            process.terminate()
            # process.join()


