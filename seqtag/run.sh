#!/usr/bin/env bash

BASE_DIR=/local/wasiahmad/workspace/projects/PrivacyIE/seqtag
#BASE_DIR=/zf18/jc6ub/Project/PrivacyIE/seqtag
DATA_DIR=${BASE_DIR}/data

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
TASK=${2:-entity}
MODEL_TYPE=${3:-bert}
USE_CRF=${4:-false} # --use_crf
SEED=${5:-1111}

if [[ $USE_CRF == true ]]; then
    USE_CRF="--use_crf"; SUFFIX="_crf";
else
    USE_CRF=""; SUFFIX="";
fi

export CUDA_VISIBLE_DEVICES=$GPU

SAVE_DIR=${BASE_DIR}/outputs/${TASK}_${MODEL_TYPE}${SUFFIX}_s${SEED}
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
    EMBED_FILE="--embed_file ${DATA_DIR}/polisis-300d-137M-subword.txt"
    USE_POSTAG="--use_postag" # USE_POSTAG=""
fi

python main.py \
    --seed $SEED \
    --task $TASK \
    --data_dir $DATA_DIR \
    --intent_label_file ${DATA_DIR}/intent_label.txt \
    --slot_label_file ${DATA_DIR}/${TASK}_slot_label.txt \
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
    --task $TASK \
    --data_dir $DATA_DIR \
    --intent_label_file ${DATA_DIR}/intent_label.txt \
    --slot_label_file ${DATA_DIR}/${TASK}_slot_label.txt \
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
        echo "Syntax: run.sh GPU_ID TASK MODEL"
        echo
        echo "GPU_ID    A list of gpu ids, separated by comma. e.g., '0,1'"
        echo "TASK      Name of the task. choices: [entity|complex]"
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
