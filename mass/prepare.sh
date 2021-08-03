#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`;

DATADIR=${HOME_DIR}/data/s2s_format
DESTDIR=${CURRENT_DIR}/resources
mkdir -p $DESTDIR


function download () {

BASE_URL=https://modelrelease.blob.core.windows.net/mass

for model_size in base middle; do
    FILE=mass-${model_size}-uncased.tar.gz
    MODEL_DIR=${DESTDIR}/mass-${model_size}-uncased
    if [[ ! -d $MODEL_DIR ]]; then
        mkdir -p $MODEL_DIR
        curl -o ${DESTDIR}/$FILE ${BASE_URL}/$FILE
        tar -xvf ${DESTDIR}/$FILE -C $MODEL_DIR
        rm ${DESTDIR}/${FILE}
    fi
done

}


function preprocess () {

for SPLIT in train valid test; do
    for LANG in source target; do
        python encode.py \
            --inputs $DATADIR/$SPLIT.$LANG \
            --outputs $DESTDIR/$SPLIT.tok.$LANG \
            --max_len 510 \
            --workers 60;
    done
done

}


function process () {

SIZE=$1
DICT_FILE=${DESTDIR}/mass-${SIZE}-uncased/dict.txt # dict.txt

fairseq-preprocess \
--user-dir mass \
--task masked_s2s \
--source-lang "source" \
--target-lang "target" \
--trainpref $DESTDIR/train.tok \
--validpref $DESTDIR/valid.tok \
--testpref $DESTDIR/test.tok \
--destdir $DESTDIR/${SIZE}-binary/ \
--workers 60 \
--srcdict $DICT_FILE \
--tgtdict $DICT_FILE;

}


download
preprocess
process base
process middle
