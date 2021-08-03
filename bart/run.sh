#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`

DATA_DIR=${CURRENT_DIR}/resources
LABEL_DIR=${HOME_DIR}/data/policyie

GPU=${1:-0}
MODEL_SIZE=${2:-base}
SEED=${3:-1111}

export CUDA_VISIBLE_DEVICES=$GPU
MAX_STEPS=2500

if [[ $MODEL_SIZE == base ]]; then
    BATCH_SIZE=4
    UPDATE_FREQ=4
    WARMUP_STEPS=100
    LR=3e-05
else
    BATCH_SIZE=1
    UPDATE_FREQ=16
    WARMUP_STEPS=250
    LR=1e-05
fi

ARCH=bart_${MODEL_SIZE}
BART_PATH=${DATA_DIR}/bart.${MODEL_SIZE}/model.pt

IFS=',' read -a GPU_IDS <<< $1
NUM_GPUS=${#GPU_IDS[@]}
EFFECT_BSZ=$((UPDATE_FREQ*BATCH_SIZE*NUM_GPUS))

DIR_SUFFIX=${ARCH}-lr${LR}-ms${MAX_STEPS}-ws${WARMUP_STEPS}-bsz${EFFECT_BSZ}-s${SEED}
SAVE_DIR=${CURRENT_DIR}/outputs/${DIR_SUFFIX}
mkdir -p $SAVE_DIR
DATA_PATH=${DATA_DIR}/${MODEL_SIZE}-binary


function train () {

fairseq-train $DATA_PATH \
--seed $SEED \
--arch $ARCH \
--save-dir $SAVE_DIR \
--restore-file $BART_PATH \
--batch-size $BATCH_SIZE \
--task translation \
--truncate-source \
--max-source-positions 1024 \
--max-target-positions 1024 \
--source-lang source \
--target-lang target \
--layernorm-embedding \
--share-all-embeddings \
--share-decoder-input-output-embed \
--reset-optimizer --reset-dataloader --reset-meters \
--required-batch-size-multiple 1 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--dropout 0.1 --attention-dropout 0.1 \
--weight-decay 0.01 --optimizer adam \
--adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
--clip-norm 0.1 \
--lr-scheduler polynomial_decay --lr $LR \
--max-update $MAX_STEPS --warmup-updates $WARMUP_STEPS \
--update-freq $UPDATE_FREQ \
--skip-invalid-size-inputs-valid-test \
--no-epoch-checkpoints --patience 5 \
--find-unused-parameters \
--ddp-backend=no_c10d 2>&1 | tee $SAVE_DIR/train_out.log;

}


function generate () {

python decode.py \
--seed $SEED \
--model_type bart \
--data_name_or_path $DATA_PATH \
--data_dir $DATA_DIR \
--checkpoint_dir $SAVE_DIR \
--checkpoint_file checkpoint_best.pt \
--output_file $SAVE_DIR/predictions.txt \
--batch_size $BATCH_SIZE \
--beam 1 \
--min_len 1 \
--lenpen 1.0 \
--max_len_b 256;

}


function evaluate () {

python evaluate.py \
--references $DATA_DIR/test.target \
--hypotheses $SAVE_DIR/predictions.txt \
--intent_label $LABEL_DIR/intent_label.txt \
--type_I_arg $LABEL_DIR/type_I_slot_label.txt \
--type_II_arg $LABEL_DIR/type_II_slot_label.txt \
--output_file $SAVE_DIR/eval_results.txt;

}


while getopts ":h" option; do
    case $option in
      h) # display Help
        echo
        echo "Syntax: run.sh GPU_ID MODEL_SIZE SEED"
        echo
        echo "GPU_ID        A list of gpu ids, separated by comma. e.g., '0,1,2'"
        echo "MODEL_SIZE    Available choices: base, large."
        echo "SEED          Seed value."
        echo
        exit;;
    esac
done


train
generate
evaluate