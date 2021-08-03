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

BATCH_SIZE=4
UPDATE_FREQ=4

if [[ $MODEL_SIZE == base ]]; then
    WARMUP_STEPS=100
    LR=3e-05
else
    WARMUP_STEPS=250
    LR=1e-05
fi

ARCH=transformer_mass_$MODEL_SIZE
MODEL_PATH=${DATA_DIR}/mass-$MODEL_SIZE-uncased/mass-$MODEL_SIZE-uncased.pt

IFS=',' read -a GPU_IDS <<< $1
NUM_GPUS=${#GPU_IDS[@]}
EFFECT_BSZ=$((UPDATE_FREQ*BATCH_SIZE*NUM_GPUS))

DIR_SUFFIX=${ARCH}-lr${LR}-ms${MAX_STEPS}-ws${WARMUP_STEPS}-bsz${EFFECT_BSZ}-s${SEED}
SAVE_DIR=${CURRENT_DIR}/outputs/${DIR_SUFFIX}
mkdir -p $SAVE_DIR
DATA_PATH=${DATA_DIR}/${MODEL_SIZE}-binary


function train () {

fairseq-train $DATA_PATH \
--user-dir mass \
--seed $SEED \
--task translation_mass --arch $ARCH \
--truncate-source \
--source-lang source \
--target-lang target \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--clip-norm 0.0 --lr 5e-4 --min-lr 1e-09 \
--lr-scheduler inverse_sqrt \
--warmup-init-lr 1e-07 \
--batch-size $BATCH_SIZE \
--max-update $MAX_STEPS \
--warmup-updates $WARMUP_STEPS \
--update-freq $UPDATE_FREQ \
--weight-decay 0.0 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--ddp-backend=no_c10d \
--max-source-positions 512 \
--max-target-positions 512 \
--skip-invalid-size-inputs-valid-test \
--no-epoch-checkpoints --patience 5 \
--find-unused-parameters \
--load-from-pretrained-model $MODEL_PATH \
--save-dir $SAVE_DIR 2>&1 | tee $SAVE_DIR/train_out.log;

}


function generate () {

fairseq-generate $DATA_PATH \
--seed $SEED \
--path ${SAVE_DIR}/checkpoint_best.pt \
--user-dir mass \
--task translation_mass \
--log-format json \
--batch-size 64 \
--beam 1 \
--min-len 1 \
--max-len-b 256 2>&1 | tee $SAVE_DIR/decode_out.log;

grep ^S $SAVE_DIR/decode_out.log | cut -f1 > $SAVE_DIR/ids.txt;
grep ^H $SAVE_DIR/decode_out.log | cut -f3- > $SAVE_DIR/hypotheses.txt;

python postprocess.py \
--index_file $SAVE_DIR/ids.txt \
--in_file $SAVE_DIR/hypotheses.txt \
--out_file $SAVE_DIR/predictions.txt;

rm $SAVE_DIR/ids.txt && rm $SAVE_DIR/hypotheses.txt;

}


function evaluate () {

python evaluate.py \
--references $DATA_DIR/test.target \
--hypotheses $SAVE_DIR/predictions.txt \
--intent_label $LABEL_DIR/intent_label.txt \
--entity_arg $LABEL_DIR/entity_slot_label.txt \
--complex_arg $LABEL_DIR/complex_slot_label.txt \
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