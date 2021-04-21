# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)


. path.sh
. cmd.sh
echo "$0 $@"

egs=$1
njobs=$2
data=$3
batch_size=$4

if [ ! -f  $egs/done_split_egs ] ; then
    python3 utils/ranges2egs_ranges.py $egs/temp $egs/
    realpath $egs/egs_range.* > $egs/egs_range_split.scp
    egs_split=$(for n in `seq $njobs`; do echo $egs/split${njobs}/$n/egs.scp;done)
    directories=$(for n in `seq $njobs`; do echo $egs//split${njobs}/$n;done)
    if ! mkdir -p $directories >&/dev/null; then
      for n in `seq $njobs`; do
        mkdir -p $egs/split${njobs}/$n
      done
    fi

    utils/split_scp.pl $egs/egs_range_split.scp $egs_split
    $train_cmd JOB=1:$njobs $egs/log/tfrecords.JOB.log \
        speaker_feature_reader.sh $data $egs/tfrecords $batch_size $egs/split${njobs}/JOB/egs.scp || exit 1;
    touch $egs/done_split_egs
fi

