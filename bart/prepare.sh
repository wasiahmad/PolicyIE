#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`

DATADIR=${HOME_DIR}/data/s2s_format
DESTDIR=${CURRENT_DIR}/resources
mkdir -p $DESTDIR


function download () {

FILE=bart.large.tar.gz
if [[ ! -d ${DESTDIR}/bart.large ]]; then
    curl -o ${DESTDIR}/${FILE} https://dl.fbaipublicfiles.com/fairseq/models/${FILE}
    tar -xvzf ${DESTDIR}/${FILE} -C ${DESTDIR}
    rm ${DESTDIR}/${FILE}
fi

FILE=bart.base.tar.gz
if [[ ! -d ${DESTDIR}/bart.base ]]; then
    curl -o ${DESTDIR}/${FILE} https://dl.fbaipublicfiles.com/fairseq/models/${FILE}
    tar -xvzf ${DESTDIR}/${FILE} -C ${DESTDIR}
    rm ${DESTDIR}/${FILE}
fi

for filename in "encoder.json" "vocab.bpe"; do
    if [[ ! -f ${DESTDIR}/${filename} ]]; then
        curl -o ${DESTDIR}/${filename} https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/${filename}
    fi
done

}


function bpe_preprocess () {

for SPLIT in train valid test; do
    for LANG in source target; do
        python encode.py \
            --encoder-json ${DESTDIR}/encoder.json \
            --vocab-bpe ${DESTDIR}/vocab.bpe \
            --inputs $DATADIR/$SPLIT.$LANG \
            --outputs $DESTDIR/$SPLIT.bpe.$LANG \
            --max_len 510 \
            --workers 60;
    done
done

}


function process () {

SIZE=$1
DICT_FILE=${DESTDIR}/bart.${SIZE}/dict.txt # dict.txt

fairseq-preprocess \
--source-lang source \
--target-lang target \
--trainpref $DESTDIR/train.bpe \
--validpref $DESTDIR/valid.bpe \
--testpref $DESTDIR/test.bpe \
--destdir $DESTDIR/$SIZE-binary \
--workers 60 \
--srcdict $DICT_FILE \
--tgtdict $DICT_FILE;

}


download
bpe_preprocess
process base
process large