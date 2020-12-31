#!/usr/bin/env bash

SRCDIR=/local/wasiahmad/workspace/projects/PrivacyIE/seqtag/data
#SRCDIR=/zf18/jc6ub/Project/PrivacyIE/seqtag/data

mkdir -p $SRCDIR
mkdir -p $SRCDIR/train
mkdir -p $SRCDIR/valid
mkdir -p $SRCDIR/test


function download () {
    FILE=${SRCDIR}/polisis-300d-137M-subword.txt
    if [[ ! -f $FILE ]]; then
        fileid="1EIwu1ahCmoHkAIpnrG-fmosbu3qsCbda"
        baseurl="https://drive.google.com/uc?export=download"
        curl -c ./cookie -s -L "${baseurl}&id=${fileid}" > /dev/null
        curl -Lb ./cookie "${baseurl}&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
        rm cookie
    fi
}

cp ../data/within_sentence_annot/train/seq.in $SRCDIR/temp.seq
cp ../data/within_sentence_annot/train/seq_entity.out $SRCDIR/temp.seq_entity
cp ../data/within_sentence_annot/train/seq_complex.out $SRCDIR/temp.seq_complex
cp ../data/within_sentence_annot/train/label $SRCDIR/temp.label
cp ../data/within_sentence_annot/train/pos_tags.out $SRCDIR/temp.pos_tags

# valid
tail -100 $SRCDIR/temp.seq > $SRCDIR/valid/seq.in
tail -100 $SRCDIR/temp.seq_entity > $SRCDIR/valid/seq_entity.out
tail -100 $SRCDIR/temp.seq_complex > $SRCDIR/valid/seq_complex.out
tail -100 $SRCDIR/temp.label > $SRCDIR/valid/label
tail -100 $SRCDIR/temp.pos_tags > $SRCDIR/valid/pos_tags.out

# train
head -n -100 $SRCDIR/temp.seq > $SRCDIR/train/seq.in
head -n -100 $SRCDIR/temp.seq_entity > $SRCDIR/train/seq_entity.out
head -n -100 $SRCDIR/temp.seq_complex > $SRCDIR/train/seq_complex.out
head -n -100 $SRCDIR/temp.label > $SRCDIR/train/label
head -n -100 $SRCDIR/temp.pos_tags > $SRCDIR/train/pos_tags.out

rm $SRCDIR/temp.seq && rm $SRCDIR/temp.label
rm $SRCDIR/temp.seq_entity && rm $SRCDIR/temp.seq_complex
rm $SRCDIR/temp.pos_tags

# test files
cp ../data/within_sentence_annot/test/seq.in $SRCDIR/test/seq.in
cp ../data/within_sentence_annot/test/seq_entity.out $SRCDIR/test/seq_entity.out
cp ../data/within_sentence_annot/test/seq_complex.out $SRCDIR/test/seq_complex.out
cp ../data/within_sentence_annot/test/label $SRCDIR/test/label
cp ../data/within_sentence_annot/test/pos_tags.out $SRCDIR/test/pos_tags.out

# label files
cp ../data/within_sentence_annot/complex_slot_label.txt $SRCDIR/complex_slot_label.txt
cp ../data/within_sentence_annot/entity_slot_label.txt $SRCDIR/entity_slot_label.txt
cp ../data/within_sentence_annot/intent_label.txt $SRCDIR/intent_label.txt
cp ../data/within_sentence_annot/postag_label.txt $SRCDIR/postag_label.txt


download
python vocab.py \
--data_dir ${SRCDIR} \
--vocab_file ${SRCDIR}/vocab.txt;
