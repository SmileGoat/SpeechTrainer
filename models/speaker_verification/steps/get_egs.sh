# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

cmd=run.pl
# each archive has data-chunks off length randomly chosen between
# $min_frames_per_eg and $max_frames_per_eg.
min_frames_per_chunk=40
max_frames_per_chunk=40
num_train_archives=3

stage=0

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <data> <egs-dir>"
  echo " e.g.: $0 data/train exp/xvector_a/egs"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl;utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --min-frames-per-eg <#frames;50>                 # The minimum number of frames per chunk that we dump"
  echo "  --max-frames-per-eg <#frames;200>                # The maximum number of frames per chunk that we dump"
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"

  exit 1;
fi

data=$1
dir=$2

for f in $data/utt2num_frames $data/feats.scp ; do
  [ ! -f $f ] && echo "$0: expected file $f" && exit 1;
done

feat_dim=$(feat-to-dim scp:$data/feats.scp -) || exit 1

mkdir -p $dir/info $dir/temp
temp=$dir/temp

echo $feat_dim > $dir/info/feat_dim
echo '0' > $dir/info/left_context
# The examples have at least min_frames_per_chunk right context.
echo $min_frames_per_chunk > $dir/info/right_context
echo '1' > $dir/info/frames_per_eg
cp $data/utt2num_frames $dir/temp/utt2num_frames

if [ $stage -le 0 ]; then
  echo "$0: Preparing train lists"
  # Create a mapping from utterance to speaker ID (an integer)
  awk -v id=0 '{print $1, id++}' $data/spk2utt > $temp/spk2int
  utils/sym2int.pl -f 2 $temp/spk2int $data/utt2spk > $temp/utt2int
fi

num_pdfs=$(awk '{print $2}' $temp/utt2int | sort | uniq -c | wc -l)

if [ $stage -le 2 ]; then
  echo "$0: Allocating training examples"
  echo $min_frames_per_chunk
  echo $max_frames_per_chunk
  echo $num_train_archives

  $cmd $dir/log/allocate_examples_train.log \
    local/allocate_egs.py \
      --min-frames-per-chunk=$min_frames_per_chunk \
      --max-frames-per-chunk=$max_frames_per_chunk \
      --num-archives=$num_train_archives \
      --utt2len-filename=$dir/temp/utt2num_frames \
      --utt2int-filename=$dir/temp/utt2int --egs-dir=$dir  || exit 1
fi
