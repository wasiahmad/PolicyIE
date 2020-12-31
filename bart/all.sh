#!/usr/bin/env bash

declare -A DIR_PREFIX
DIR_PREFIX['bart_base']='lr3e-05-ms2500-ws100-bsz16'
DIR_PREFIX['bart_large']='lr1e-05-ms2500-ws250-bsz16'


function get_results () {

BASE_DIR=/local/wasiahmad/workspace/projects/PrivacyIE/seq2seq/outputs
for model in bart_base bart_large; do
    python result.py \
        --model $model \
        --base_dir $BASE_DIR \
        --dir_prefix ${DIR_PREFIX[${model}]} \
        --seeds 1111 2222 3333 4444 5555;
done

}

get_results
