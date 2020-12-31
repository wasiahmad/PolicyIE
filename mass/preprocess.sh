#!/usr/bin/env bash

SRCDIR=/local/wasiahmad/workspace/projects/PrivacyIE/mass/data

function download () {

BASE_URL=https://modelrelease.blob.core.windows.net/mass

for model_size in base middle; do
    FILE=mass-${model_size}-uncased.tar.gz
    MODEL_DIR=${SRCDIR}/mass-${model_size}-uncased
    if [[ ! -d $MODEL_DIR ]]; then
        mkdir -p $MODEL_DIR
        curl -o ${SRCDIR}/$FILE ${BASE_URL}/$FILE
        tar -xvf ${SRCDIR}/$FILE -C $MODEL_DIR
        rm ${SRCDIR}/${FILE}
    fi
done

}

function preprocess () {

for SPLIT in train valid test; do
    for LANG in source target; do
        python encode.py \
            --inputs "$SRCDIR/$SPLIT.$LANG" \
            --outputs "$SRCDIR/$SPLIT.tok.$LANG" \
            --max_len 510 \
            --workers 60; \
    done
done

}

function process () {

SIZE=$1
DICT_FILE=${SRCDIR}/mass-${SIZE}-uncased/dict.txt # dict.txt

fairseq-preprocess \
--user-dir mass \
--task masked_s2s \
--source-lang "source" \
--target-lang "target" \
--trainpref $SRCDIR/train.tok \
--validpref $SRCDIR/valid.tok \
--testpref $SRCDIR/test.tok \
--destdir $SRCDIR/${SIZE}-binary/ \
--workers 60 \
--srcdict $DICT_FILE \
--tgtdict $DICT_FILE;

}

function prepare () {

mkdir -p $SRCDIR
cp ../data/within_sentence_annot/train/seq.in $SRCDIR/temp.source
cp ../data/within_sentence_annot/train/seq.out $SRCDIR/temp.target
cp ../data/within_sentence_annot/test/seq.in $SRCDIR/test.source
cp ../data/within_sentence_annot/test/seq.out $SRCDIR/test.target

tail -100 $SRCDIR/temp.source > $SRCDIR/valid.source
tail -100 $SRCDIR/temp.target > $SRCDIR/valid.target
head -n -100 $SRCDIR/temp.source > $SRCDIR/train.source
head -n -100 $SRCDIR/temp.target > $SRCDIR/train.target
rm $SRCDIR/temp.source && rm $SRCDIR/temp.target

}

prepare
download
preprocess
process base
process middle
