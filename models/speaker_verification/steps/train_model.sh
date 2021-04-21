# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

echo "$0 $@"

[ -f ./path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
    echo "Usage: local/extract_embeddings.sh --nj 50 <nnet-dir> <data> <output-dir>"
    echo "..."
fi

train_egs=$1
dev_egs=$2
exp_dir=$3

for f in $dev_egs/egs_tfrecords.scp $train_egs/egs_tfrecord_1.scp; do
  [ ! -f $f ] && echo "$0: Error: no such file $f" && exit 1;
done

local/train.py --egs.train_dir=$train_egs \
    --egs.dev_dir=$dev_egs \
    --trainer.srand=1000 \
    --trainer.num-epochs=10 \
		--trainer.num-jobs-final=1 \
		--trainer.num-jobs-initial=8 \
		--trainer.model-dir=$exp_dir \
		--trainer.initial-effective-lrate=0.01 \
		--trainer.final-effective-lrate=0.0001 \
		--trainer.num-archives=3
