#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`

AVAILABLE_MODEL_CHOICES=(
    bert
    distilbert
    albert
    roberta
    transformer
    rnn
    feature
)

GPU=${1:-0}
SLOT_TYPE=${2:-type_I}
MODEL_TYPE=${3:-bert}
USE_CRF=${4:-false}
SEED=${5:-1111}

if [[ $USE_CRF == true ]]; then
    USE_CRF="--use_crf"; SUFFIX="_crf";
else
    USE_CRF=""; SUFFIX="";
fi

export CUDA_VISIBLE_DEVICES=$GPU

EMBED_DIR=${HOME_DIR}/data
DATA_DIR=${HOME_DIR}/data/bio_format
SAVE_DIR=${CURRENT_DIR}/outputs/${SLOT_TYPE}_${MODEL_TYPE}${SUFFIX}_s${SEED}
mkdir -p $SAVE_DIR


function train () {

LR=3e-5
NUM_EPOCHS=10
EMBED_FILE=""

if [[ $MODEL_TYPE == *'transformer'* ]]; then
    LR=1e-4
    NUM_EPOCHS=20
elif [[ $MODEL_TYPE == *'rnn'* ]]; then
    LR=1e-3
    NUM_EPOCHS=20
fi
if [[ $MODEL_TYPE == 'feature' ]]; then
    LR=1e-3
    NUM_EPOCHS=20
    EMBED_FILE="--embed_file ${EMBED_DIR}/polisis-300d-137M-subword.txt"
    USE_POSTAG="--use_postag"
fi

python main.py \
    --seed $SEED \
    --task $SLOT_TYPE \
    --data_dir $DATA_DIR \
    --intent_label_file ${DATA_DIR}/intent_label.txt \
    --slot_label_file ${DATA_DIR}/${SLOT_TYPE}_slot_label.txt \
    --postag_label_file ${DATA_DIR}/postag_label.txt \
    --model_type $MODEL_TYPE \
    --model_dir $SAVE_DIR \
    --learning_rate $LR \
    --save_steps 100 \
    --logging_steps 100 \
    --max_seq_len 384 \
    --eval_patience 10 \
    --num_train_epochs $NUM_EPOCHS \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --do_train $EMBED_FILE \
    $USE_POSTAG \
    --do_eval $USE_CRF 2>&1 | tee $SAVE_DIR/train_out.log;

}


function evaluate () {

python main.py \
    --task $SLOT_TYPE \
    --data_dir $DATA_DIR \
    --intent_label_file ${DATA_DIR}/intent_label.txt \
    --slot_label_file ${DATA_DIR}/${SLOT_TYPE}_slot_label.txt \
    --postag_label_file ${DATA_DIR}/postag_label.txt \
    --model_type $MODEL_TYPE \
    --max_seq_len 384 \
    --eval_batch_size 16 \
    --model_dir $SAVE_DIR \
    --do_eval $USE_CRF 2>&1 | tee $SAVE_DIR/eval_out.log;

}


while getopts ":h" option; do
    case $option in
      h) # display Help
        echo
        echo "Syntax: run.sh GPU_ID SLOT_TYPE MODEL"
        echo
        echo "GPU_ID    A list of gpu ids, separated by comma. e.g., '0,1'"
        echo "SLOT_TYPE Name of the task. choices: [type_I|type_II]"
        echo "MODEL     Model name; choices: [$(IFS=\| ; echo "${AVAILABLE_MODEL_CHOICES[*]}")]"
        echo
        exit;;
    esac
done

if [[ ! " ${AVAILABLE_MODEL_CHOICES[@]} " =~ " ${MODEL_TYPE} " ]]; then
    echo "Invalid model choice. Available choices: [$(IFS=\| ; echo "${AVAILABLE_MODEL_CHOICES[*]}")]"
    exit 1
fi


train
evaluate