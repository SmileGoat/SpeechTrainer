# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

import os, argparse, sys, random, math

import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Writes ranges.*, and archive_chunk_lengths files "
                                 "in preparation for dumping tfrecords for xvector training.",
                                 epilog="Called by local/get_egs.sh")
    parser.add_argument("--prefix", type=str, default="",
                   help="Adds a prefix to the output files. This is used to distinguish between the train "
                   "and diagnostic files.")
    parser.add_argument("--min-frames-per-chunk", type=int, default=50,
                    help="Minimum number of frames-per-chunk used for any archive")
    parser.add_argument("--max-frames-per-chunk", type=int, default=300,
                    help="Maximum number of frames-per-chunk used for any archive")
    parser.add_argument("--random-chunk-length", type=str,
                    help="If true, randomly pick a chunk length in [min-frames-per-chunk, max-frames-per-chunk]."
                    "If false, the chunk length varies from min-frames-per-chunk to max-frames-per-chunk"
                    "according to a geometric sequence.",
                    default="true", choices = ["false", "true"])
    parser.add_argument("--num-archives", type=int, default=200,
                    help="num archives")
    parser.add_argument("--seed", type=int, default=123,
                    help="Seed for random number generator")

    # now the positional arguments
    parser.add_argument("--utt2len-filename", type=str, required=True,
                    help="utt2len file of the features to be used as input (format is: "
                    "<utterance-id> <num-frames>)");
    parser.add_argument("--utt2int-filename", type=str, required=True,
                    help="utt2int file of the features to be used as input (format is: "
                    "<utterance-id> <id>)");
    parser.add_argument("--egs-dir", type=str, required=True,
                    help="Name of egs directory, e.g. exp/xvector_a/egs");

    print(' '.join(sys.argv), file=sys.stderr)
    print(sys.argv, file=sys.stderr)
    args = parser.parse_args()
    args = process_args(args)
    return args

def process_args(args):
    if not os.path.exists(args.utt2int_filename):
        raise Exception("This script expects --utt2int-filename to exist")
    if not os.path.exists(args.utt2len_filename):
        raise Exception("This script expects --utt2len-filename to exist")
    if args.min_frames_per_chunk <= 1:
        raise Exception("--min-frames-per-chunk is invalid.")
    if args.max_frames_per_chunk < args.min_frames_per_chunk:
        raise Exception("--max-frames-per-chunk is invalid.")
    return args

# Create utt2len
def get_utt2len(utt2len_filename):
    utt2len = {}
    f = open(utt2len_filename, "r")
    if f is None:
        sys.exit("Error opening utt2len file " + str(utt2len_filename))
    utt_ids = []
    lengths = []
    for line in f:
        tokens = line.split()
        if len(tokens) != 2:
            sys.exit("bad line in utt2len file " + line)
        utt2len[tokens[0]] = int(tokens[1])
    f.close()
    return utt2len
    # Done utt2len

# Handle utt2int, create spk2utt, spks
def get_labels(utt2int_filename):
    f = open(utt2int_filename, "r")
    if f is None:
        sys.exit("Error opening utt2int file " + str(utt2int_filename))
    spk2utt = {}
    utt2spk = {}
    for line in f:
        tokens = line.split()
        if len(tokens) != 2:
            sys.exit("bad line in utt2int file " + line)
        spk = int(tokens[1])
        utt = tokens[0]
        utt2spk[utt] = spk
        if spk not in spk2utt:
            spk2utt[spk] = [utt]
        else:
            spk2utt[spk].append(utt)
    spks = spk2utt.keys()
    f.close()
    return spks, spk2utt, utt2spk
    # Done utt2int

# this function returns a random integer utterance index, limited to utterances
# above a minimum length in frames, with probability proportional to its length.
def get_random_utt(spkr, spk2utt, min_length):
    this_utts = spk2utt[spkr]
    this_num_utts = len(this_utts)
    i = random.randint(0, this_num_utts-1)
    utt = this_utts[i]
    return utt

def random_chunk_length(min_frames_per_chunk, max_frames_per_chunk):
    ans = random.randint(min_frames_per_chunk, max_frames_per_chunk)
    return ans

# This function returns an integer in the range
# [min-frames-per-chunk, max-frames-per-chunk] according to a geometric
# sequence. For example, suppose min-frames-per-chunk is 50,
# max-frames-per-chunk is 200, and args.num_archives is 3. Then the
# lengths for archives 0, 1, and 2 will be 50, 100, and 200.
def deterministic_chunk_length(archive_id, num_archives, min_frames_per_chunk, max_frames_per_chunk):
  if max_frames_per_chunk == min_frames_per_chunk:
    return max_frames_per_chunk
  elif num_archives == 1:
    return int(max_frames_per_chunk)
  else:
    return int(math.pow(float(max_frames_per_chunk)/
                     min_frames_per_chunk, float(archive_id) /
                     (num_archives-1)) * min_frames_per_chunk + 0.5)

def get_spk_egs(utt, spk, utt_len, chunk_len):
    result = []
    num_chunks = int(np.ceil(float(utt_len - chunk_len) / (chunk_len / 2)))
    if (num_chunks <= 0): return result
    
    for i in range(num_chunks):
        start = int (i * (chunk_len / 2))
        this_chunk_start = start if start + chunk_len < utt_len else utt_len - chunk_len
        result.append((utt, this_chunk_start, chunk_len, spk))

    return result


def main():
    args = get_args()
    if not os.path.exists(args.egs_dir + "/temp"):
        os.makedirs(args.egs_dir + "/temp")
    random.seed(args.seed)
    utt2len = get_utt2len(args.utt2len_filename)
    spks, spk2utt, utt2spk = get_labels(args.utt2int_filename)

    archive_chunnk_lengths = []
    all_egs = []

    for archive_idx in range(args.num_archives):
        print("Processing archive {0}".format(archive_idx + 1))
        this_egs = []
        f = open(args.egs_dir + "/temp/" + "ranges." + str(archive_idx), "w")
        if args.random_chunk_length == "true":
            chunk_len = random_chunk_length(args.min_frames_per_chunk, args.max_frames_per_chunk)
        else:
            chunk_len = deterministic_chunk_length(archive_idx, args.num_archives, args.min_frames_per_chunk, args.max_frames_per_chunk)
        for spk in spks:
            utt_nums = len(spk2utt[spk])
            this_spk_utt = spk2utt[spk][archive_idx % utt_nums]
            utt_len = utt2len.get(this_spk_utt)
            if utt_len == None:
                print("{0} is not in utt2num_frames".format(this_spk_utt))
                continue
            this_spk_egs = get_spk_egs(this_spk_utt, spk, utt_len, chunk_len)
            this_egs = this_egs + this_spk_egs
        random.shuffle(this_egs)
        for elem in this_egs:
            print("{0} {1} {2} {3} {4} {5}".format(elem[0],
                                           0,
                                           archive_idx + 1,
                                           elem[1],
                                           elem[2],
                                           elem[3]),
                                           file=f)
        f.close()

if __name__ == "__main__":
    main()
