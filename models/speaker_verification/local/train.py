# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)


import sys
import os
import argparse
import tensorflow as tf

sys.path.insert(0, '../..')

from models.trainer import train

def get_args():
    parser = argparse.ArgumentParser(description="speaker recognition training.",
                                 epilog="Called by local/train.py")

    parser.add_argument("--egs.train_dir", type=str, dest='train_egs_dir',
                                 default=None,
                                 help="""Directory with tfrecords""")
    parser.add_argument("--egs.dev_dir", type=str, dest='dev_egs_dir',
                                 default=None,
                                 help="""Directory with tfrecords""")

    parser.add_argument("--trainer.srand", type=int, dest='srand',
                             default=0,
                             help="""Sets the random seed for tfrecords shuffling.
                             """)

    parser.add_argument("--trainer.num-epochs", type=float,
                             dest='num_epochs', default=10.0,
                             help="Number of epochs to train the model")

    parser.add_argument("--trainer.num-archives", type=int,
                             dest='num_archives', default=None,
                             help="Number of epochs to train the model")

    parser.add_argument("--trainer.num-jobs-final", type=int,
                             dest='num_jobs_final', default=1,
                             help="Number of jobs to train the model")

    parser.add_argument("--trainer.num-jobs-initial", type=int,
                             dest='num_jobs_initial', default=8,
                             help="Number of jobs to train the model at begining")

    parser.add_argument("--trainer.model-dir", type=str,
                             dest='model_dir', default=None,
                             help="")

    parser.add_argument("--trainer.num-jobs-step", type=int,
                             dest='num_jobs_step', default=1,
                             help="""Number of jobs increment, when exceeds this number. 
                                     For example, if N=3, the number of jobs
                                     may progress as 1, 2, 3, 6, 9...""")

    parser.add_argument("--trainer.initial-effective-lrate", type=float,
                             dest='initial_effective_lrate', default=None,
                             help="")
    parser.add_argument("--trainer.final-effective-lrate", type=float,
                             dest='final_effective_lrate', default=None,
                             help="")
    # General options
    parser.add_argument("--stage", type=int, default=-4,                           
                             help="Specifies the stage of the experiment "
                             "to execution from")
    parser.add_argument("--use-gpu", type=str,
                             choices=["true", "false", "yes", "no", "wait"],
                             help="Use GPU for training. "
                             "Note 'true' and 'false' are deprecated.",
                             default="yes")

    print(' '.join(sys.argv), file=sys.stderr)
    print(sys.argv, file=sys.stderr)
    args = parser.parse_args()
    args = process_args(args)
    return args

def process_args(args):
    if not os.path.exists(args.train_egs_dir):
        raise Exception("This script expects 0 to exist")
    if not os.path.exists(args.dev_egs_dir):
        raise Exception("This script expects 1 to exist")
    if not os.path.exists(args.model_dir):
        raise Exception("This script expects 2 to exist")
    return args

def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    args = get_args()
    train(args.num_jobs_final,
          args.num_jobs_initial,
          args.num_archives,
          args.num_epochs,
          args.model_dir,
          args.num_jobs_step,
          args.srand,
          args.initial_effective_lrate,
          args.final_effective_lrate,
          args.train_egs_dir, args.dev_egs_dir)

if __name__ == "__main__":
    main()
