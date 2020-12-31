#!/usr/bin/env bash

declare -A DIR_PREFIX
DIR_PREFIX['unilm1-base']='lr1e-4-ms2500-ws250-bsz16'
DIR_PREFIX['unilm1-large']='lr1e-4-ms2500-ws250-bsz16'
DIR_PREFIX['unilm2']='lr1e-4-ms2500-ws250-bsz16'
DIR_PREFIX['minilm']='lr1e-4-ms2500-ws250-bsz16'


function get_results () {

BASE_DIR=/local/wasiahmad/workspace/projects/PrivacyIE/unilm/outputs
for model in minilm unilm1-base unilm2; do
    python result.py \
        --model $model \
        --base_dir $BASE_DIR \
        --dir_prefix ${DIR_PREFIX[${model}]} \
        --seeds 1111 2222 3333 4444 5555;
done

}

get_results
