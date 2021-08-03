#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`;

function download () {
    FILE=${CURRENT_DIR}/combined_policies.zip
    if [[ ! -f $FILE ]]; then
        fileid="1qUKjO8hw384jxdgqa-P1JhSeIBt9XwNY"
        baseurl="https://drive.google.com/uc?export=download"
        curl -c ./cookie -s -L "${baseurl}&id=${fileid}" > /dev/null
        curl -Lb ./cookie "${baseurl}&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
        unzip $FILE -d $CURRENT_DIR
        rm cookie && $CURRENT_DIR/__MACOSX
    fi
}

function spm_train () {

python spm_train.py \
    --input_file ${CURRENT_DIR}/combined_policies.txt \
    --vocab_size 10000;

}

function spm_tokenize () {

python encode.py \
    --model-file sentencepiece.bpe.model \
    --inputs ${CURRENT_DIR}/combined_policies.txt \
    --outputs ${CURRENT_DIR}/combined_policies.spm \
    --max_len 2048 \
    --workers 60;

}

function fasttext_train () {

python fast_train.py \
    --input_file ${CURRENT_DIR}/combined_policies.spm \
    --emb_size 128;

}

download
spm_train && spm_tokenize
fasttext_train