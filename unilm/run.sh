#!/usr/bin/env bash

BASE_DIR=/local/wasiahmad/workspace/projects/PrivacyIE/unilm
DATA_DIR=${BASE_DIR}/data

#https://github.com/microsoft/unilm/tree/master/s2s-ft#pre-trained-models
AVAILABLE_MODEL_CHOICES=(
    unilm1-base
    unilm1-large
    unilm2
    minilm
)

while getopts ":h" option; do
   case $option in
      h) # display Help
        echo
        echo "Syntax: run.sh GPU_ID MODEL_NAME SEED"
        echo
        echo "GPU_ID         A list of gpu ids, separated by comma. e.g., '0,1,2'"
        echo "MODEL_NAME     Model name; choices: [$(IFS=\| ; echo "${AVAILABLE_MODEL_CHOICES[*]}")]"
        echo "SEED          Seed value."
        echo
        exit;;
   esac
done


GPU=${1:-0}
MODEL_CHOICE=${2:-"minilm"}
SEED=${3:-1111}

if [[ $MODEL_CHOICE == 'unilm1-base' ]]; then
    MODEL_TYPE=unilm
    MODEL_NAME_OR_PATH=unilm1-base-cased
elif [[ $MODEL_CHOICE == 'unilm1-large' ]]; then
    MODEL_TYPE=unilm
    MODEL_NAME_OR_PATH=unilm1-large-cased
elif [[ $MODEL_CHOICE == 'unilm2' ]]; then
    MODEL_TYPE=unilm
    MODEL_NAME_OR_PATH=unilm1.2-base-uncased
elif  [[ $MODEL_CHOICE == 'minilm' ]]; then
    MODEL_TYPE=minilm
    MODEL_NAME_OR_PATH=minilm-l12-h384-uncased
else
    echo -n "... Wrong model choice!! available choices: [$(IFS=\| ; echo "${AVAILABLE_MODEL_CHOICES[*]}")]" ;
    exit 1
fi


export CUDA_VISIBLE_DEVICES=$GPU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

PER_GPU_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LR=1e-4
NUM_WARM_STEPS=250
NUM_TRAIN_STEPS=2500

CKPT_NAME=ckpt-$NUM_TRAIN_STEPS

IFS=',' read -a GPU_IDS <<< $GPU
NUM_GPUS=${#GPU_IDS[@]}
EFFECT_BSZ=$((GRADIENT_ACCUMULATION_STEPS*PER_GPU_TRAIN_BATCH_SIZE*NUM_GPUS))

DIR_SUFFIX=${MODEL_CHOICE}-lr${LR}-ms${NUM_TRAIN_STEPS}-ws${NUM_WARM_STEPS}-bsz${EFFECT_BSZ}-s${SEED}
OUTPUT_DIR=${BASE_DIR}/outputs/${DIR_SUFFIX}
mkdir -p $OUTPUT_DIR


function train () {

# folder used to cache package dependencies
CACHE_DIR=~/.cache/torch/transformers
LOG_FILENAME=${OUTPUT_DIR}/train_log.txt

python s2s-ft/run_seq2seq.py \
--seed $SEED \
--train_file $DATA_DIR/train.json \
--output_dir ${OUTPUT_DIR} \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL_NAME_OR_PATH \
--save_steps 250 \
--do_lower_case \
--max_source_seq_length 256 \
--max_target_seq_length 256 \
--per_gpu_train_batch_size $PER_GPU_TRAIN_BATCH_SIZE \
--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
--learning_rate $LR \
--num_warmup_steps $NUM_WARM_STEPS \
--num_training_steps $NUM_TRAIN_STEPS \
--cache_dir ${CACHE_DIR} \
--workers 20 2>&1 | tee $LOG_FILENAME;

}


function decode () {

SPLIT=test
LOG_FILENAME=$OUTPUT_DIR/test_log.txt

python s2s-ft/decode_seq2seq.py \
--seed $SEED \
--model_type $MODEL_TYPE \
--tokenizer_name $MODEL_NAME_OR_PATH \
--input_file $DATA_DIR/${SPLIT}.json \
--split $SPLIT \
--do_lower_case \
--model_path $OUTPUT_DIR/$CKPT_NAME \
--max_seq_length 512 \
--max_tgt_length 256 \
--batch_size 64 \
--beam_size 1 \
--length_penalty 1.0 \
--mode s2s \
--output_file $OUTPUT_DIR/$CKPT_NAME.$SPLIT \
--workers 20 2>&1 | tee $LOG_FILENAME;

}


function evaluate () {

SPLIT=test
LABEL_DIR=../data/within_sentence_annot

python evaluate.py \
--references $DATA_DIR/test.target \
--hypotheses $OUTPUT_DIR/$CKPT_NAME.$SPLIT \
--intent_label $LABEL_DIR/intent_label.txt \
--entity_arg $LABEL_DIR/entity_slot_label.txt \
--complex_arg $LABEL_DIR/complex_slot_label.txt \
--output_file $OUTPUT_DIR/eval_results.txt;

}


train
decode
evaluate
