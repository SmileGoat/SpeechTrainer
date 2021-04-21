# Copyright (c) 2021 PeachLab. All Rights Reserved.
# Author : goat.zhou@qq.com (Yang Zhou)

. ./cmd.sh
. ./path.sh

set -e

# process data
stage=2
njobs=60
# prepare egs for training
if ! test -d utils; then
    ln -s ../../utils .
    ln -s ../../models/speaker_verification/local .
else 
    echo "utils have linked, skip"
fi

data=
train_dev=$data/train_dev
train=$data/train
dev=$data/dev
mkdir -p $train
mkdir -p $dev

if [ $stage -le 0 ] && [ ! -f $data/done_split_train_dev ]; then

    for f in $train_dev/feats.scp $train_dev/spk2utt $train_dev/utt2num_frames ; do
        [ ! -f $f ] && echo "No such file $f" && exit 1;
    done

    local/split_train_dev.py 0.1 $train_dev $train $dev

    utils/filter_scp.pl $train/feats.scp $train_dev/utt2num_frames > $train/utt2num_frames
    utils/filter_scp.pl $dev/feats.scp $train_dev/utt2num_frames > $dev/utt2num_frames

    utils/spk2utt_to_utt2spk.pl $train/spk2utt > $train/utt2spk
    utils/spk2utt_to_utt2spk.pl $dev/spk2utt > $dev/utt2spk

    touch $data/done_split_train_dev

fi

exp=$PWD/exp/tf_refactor_kaldi_tdnn_210121
egs=$exp/egs

if [ $stage -le 1 ]; then
    steps/get_egs.sh --num_train_archives 3 \
      $data/train $egs/train
    
    steps/get_egs.sh --num_train_archives 1 \
      $data/dev $egs/dev
fi

if [ $stage -le 1 ] && [ ! -f $egs/done_produce_tfrecords ] ; then

    batch_size=64
    local/produce_tfrecords_by_ranges.sh $egs/train_egs $njobs $data/train $batch_size || exit 1
    local/produce_tfrecords_by_ranges.sh $egs/dev_egs $njobs $data/dev $batch_size || exit 1
    touch $egs/done_produce_tfrecords

fi

nnet_dir=$exp/nnet
mkdir -p $nnet_dir

if [ $stage -le 2 ] ; then

   steps/train_model.sh $egs/train_egs $egs/dev_egs $nnet_dir

fi

# training plda

if [ $stage -le 3 ] ; then

    local/extract_embeddings.sh --njobs $njobs $nnet_dir $data/train $exp/xvectors_vivo_train

fi

if [ $stage -le 4 ] ; then

    local/extract_embeddings.sh --njobs $njobs $nnet_dir $data/dev $exp/xvectors_vivo_dev

fi

xvectors_dev=$exp/xvectors_vivo_dev
xvectors_train=$exp/xvectors_vivo_train
xvectors_eval=$exp/xvectors_vivo_eval
xvectors_enroll=$exp/xvectors_vivo_enroll

if [ $stage -le 5 ] ; then

    local/extract_embeddings.sh --njobs $njobs $nnet_dir $data/test/eval $exp/xvectors_vivo_eval
    local/extract_embeddings.sh --njobs $njobs $nnet_dir $data/test/enroll $exp/xvectors_vivo_enroll

fi

if [ $stage -le 6 ] ; then
    ivector-mean scp:$xvectors_train/xvectors.scp $xvectors_train/mean.vec || exit 1; 

    lda_dim=200
    $train_cmd $xvectors_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:$xvectors_train/xvectors.scp ark:- |" \
    ark:$data/train/utt2spk $xvectors_train/transform.mat || exit 1;

  # Train an out-of-domain PLDA model.
    $train_cmd $xvectors_train/log/plda.log \
    ivector-compute-plda ark:$data/train/spk2utt \
    "ark:ivector-subtract-global-mean scp:$xvectors_train/xvectors.scp ark:- | transform-vec $xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    $xvectors_train/plda || exit 1;

    $train_cmd $xvectors_dev/log/plda_adapt.log \
    ivector-adapt-plda --within-covar-scale=0.75 --between-covar-scale=0.25 \
    $xvectors_train/plda \
    "ark:ivector-subtract-global-mean scp:$xvectors_dev/xvectors.scp ark:- | transform-vec $xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    $xvectors_dev/plda_adapt || exit 1;
fi

trials=$data/test/vivo_speaker_ver.lst
xvector_enroll_dir=$exp/xvectors_vivo_enroll
xvector_eval_dir=$exp/xvectors_vivo_eval
num_utts=$data/test/num_utts.ark

if [ $stage -le 7 ] ; then
    $train_cmd $exp/scores/log/vivo_eval_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    --num-utts=ark:$num_utts \
    "ivector-copy-plda --smoothing=0.0 $xvectors_train/plda - |" \
    "ark:ivector-mean ark:$data/test/enroll/spk2utt scp:$xvector_enroll_dir/xvectors.scp ark:- | ivector-subtract-global-mean $xvectors_train/mean.vec ark:- ark:- | transform-vec $xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $xvectors_train/mean.vec scp:$xvector_eval_dir/xvectors.scp ark:- | transform-vec $xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$trials' |  awk '{print \\\$2, \\\$1}' |" $exp/scores/eval_scores || exit 1;

  python filter_scores.py $exp/scores/eval_scores $exp/scores/filter_scores
  awk '{print $3,$4}' $exp/scores/filter_scores | compute-eer - > $exp/scores/eval_eer
fi
