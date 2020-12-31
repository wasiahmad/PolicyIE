#!/usr/bin/env bash

SRCDIR=/local/wasiahmad/workspace/projects/PrivacyIE/embeddings
mkdir -p $SRCDIR

function download () {
    FILE=${SRCDIR}/combined_policies.zip
    if [[ ! -f $FILE ]]; then
        fileid="1qUKjO8hw384jxdgqa-P1JhSeIBt9XwNY"
        baseurl="https://drive.google.com/uc?export=download"
        curl -c ./cookie -s -L "${baseurl}&id=${fileid}" > /dev/null
        curl -Lb ./cookie "${baseurl}&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
        unzip $FILE -d $SRCDIR
        rm cookie && $SRCDIR/__MACOSX
    fi
}

function spm_train () {

python spm_train.py \
    --input_file ${SRCDIR}/combined_policies.txt \
    --vocab_size 10000;

}

function spm_tokenize () {

python encode.py \
    --model-file sentencepiece.bpe.model \
    --inputs ${SRCDIR}/combined_policies.txt \
    --outputs ${SRCDIR}/combined_policies.spm \
    --max_len 2048 \
    --workers 60;

}

function fasttext_train () {

python fast_train.py \
    --input_file ${SRCDIR}/combined_policies.spm \
    --emb_size 128;

}


download
spm_train && spm_tokenize
fasttext_train
