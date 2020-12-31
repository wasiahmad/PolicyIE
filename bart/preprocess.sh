#!/usr/bin/env bash

SRCDIR=/local/wasiahmad/workspace/projects/PrivacyIE/seq2seq/data

function download () {

#FILE=${SRCDIR}/glove.840B.300d.txt
#if [[ ! -f $FILE ]]; then
#    curl -o ${SRCDIR} http://nlp.stanford.edu/data/glove.840B.300d.zip
#    unzip ${SRCDIR}/glove.840B.300d.zip -d ${SRCDIR}
#    rm ${SRCDIR}/glove.840B.300d.zip
#fi

#FILE=${SRCDIR}/polisis-300d-137M-subword.txt
#if [[ ! -f $FILE ]]; then
#    fileid="1EIwu1ahCmoHkAIpnrG-fmosbu3qsCbda"
#    baseurl="https://drive.google.com/uc?export=download"
#    curl -c ./cookie -s -L "${baseurl}&id=${fileid}" > /dev/null
#    curl -Lb ./cookie "${baseurl}&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
#    rm cookie
#fi

FILE=bart.large.tar.gz
if [[ ! -d ${SRCDIR}/bart.large ]]; then
    curl -o ${SRCDIR}/${FILE} https://dl.fbaipublicfiles.com/fairseq/models/${FILE}
    tar -xvzf ${SRCDIR}/${FILE} -C ${SRCDIR}
    rm ${SRCDIR}/${FILE}
fi

FILE=bart.base.tar.gz
if [[ ! -d ${SRCDIR}/bart.base ]]; then
    curl -o ${SRCDIR}/${FILE} https://dl.fbaipublicfiles.com/fairseq/models/${FILE}
    tar -xvzf ${SRCDIR}/${FILE} -C ${SRCDIR}
    rm ${SRCDIR}/${FILE}
fi

for filename in "encoder.json" "vocab.bpe"; do
    if [[ ! -f ${SRCDIR}/${filename} ]]; then
        curl -o ${SRCDIR}/${filename} https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/${filename}
    fi
done

}


function bpe_preprocess () {

for SPLIT in train valid test; do
    for LANG in source target; do
        python encode.py \
            --encoder-json ${SRCDIR}/encoder.json \
            --vocab-bpe ${SRCDIR}/vocab.bpe \
            --inputs $SRCDIR/$SPLIT.$LANG \
            --outputs $SRCDIR/$SPLIT.bpe.$LANG \
            --max_len 510 \
            --workers 60;
    done
done

}


function process () {

SIZE=$1
DICT_FILE=${SRCDIR}/bart.${SIZE}/dict.txt # dict.txt

fairseq-preprocess \
--source-lang source \
--target-lang target \
--trainpref $SRCDIR/train.bpe \
--validpref $SRCDIR/valid.bpe \
--testpref $SRCDIR/test.bpe \
--destdir $SRCDIR/$SIZE-binary \
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
bpe_preprocess
process base
process large
