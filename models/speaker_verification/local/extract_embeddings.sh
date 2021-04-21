# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

njobs=60
cmd=run.pl
feats=

echo "$0 $@"

[ -f ./path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
    echo "Usage: local/extract_embeddings.sh --nj 50 <nnet-dir> <data> <output-dir>"
    echo "..."
fi

nnet_dir=$1
data=$2
exp_dir=$3

#for f in $nnet_dir/final.raw.meta $data/feats.scp; do
for f in $data/feats.scp; do
  [ ! -f $f ] && echo "$0: Error: no such file $f" && exit 1;
done

feats_split=$(for n in `seq $njobs`; do echo $data/split${njobs}/$n/feats.scp;done)
directories=$(for n in `seq $njobs`; do echo $data/split${njobs}/$n;done)
if ! mkdir -p $directories >&/dev/null; then
  for n in `seq $njobs`; do
    mkdir -p $data/split${njobs}/$n
  done
fi

if [ ! -f $data/split${njobs}/1/feats.scp ]; then
    split_scp.pl $data/feats.scp $feats_split
fi

logdir=$exp_dir/log
mkdir -p $logdir
embeddings=$exp_dir/embeddings
mkdir -p $embeddings
model=$nnet_dir/final.raw


feats=$data/split${njobs}/JOB/feats.scp
$cmd JOB=1:$njobs $logdir/extract_embeddings.JOB.log \
  forward.sh $feats $model ark,scp:$embeddings/xvector.JOB.ark,$embeddings/xvector.JOB.scp JOB \
  || exit 1;

cat $embeddings/xvector.*.scp > $embeddings/../xvectors.scp


