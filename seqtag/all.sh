#!/usr/bin/env bash

function get_results () {

BASE_DIR=/local/wasiahmad/workspace/projects/PrivacyIE/seqtag/outputs
for crf in False True; do
    for model in rnn transformer bert roberta; do
        python result.py \
            --model $model \
            --base_dir $BASE_DIR \
            --use_crf $crf \
            --seeds 1111 2222 3333 4444 5555;
    done
done

}

function run () {

task=$1
model=$2
use_crf=$3

nohup bash run.sh 1 $task $model $use_crf 1111 &
nohup bash run.sh 2 $task $model $use_crf 2222 &
nohup bash run.sh 3 $task $model $use_crf 3333 &
nohup bash run.sh 4 $task $model $use_crf 4444 &
nohup bash run.sh 5 $task $model $use_crf 5555 &

}

#run $1 $2 $3
get_results
