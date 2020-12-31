#!/usr/bin/env bash

declare -A DIR_PREFIX
DIR_PREFIX['transformer_mass_base']='lr3e-05-ms2500-ws100-bsz16'
DIR_PREFIX['transformer_mass_middle']='lr1e-05-ms2500-ws250-bsz16'

function get_results () {

BASE_DIR=/local/wasiahmad/workspace/projects/PrivacyIE/mass/outputs
for model in transformer_mass_base transformer_mass_middle; do
    python result.py \
        --model $model \
        --base_dir $BASE_DIR \
        --dir_prefix ${DIR_PREFIX[${model}]} \
        --seeds 1111 2222 3333 4444 5555;
done

}

get_results
