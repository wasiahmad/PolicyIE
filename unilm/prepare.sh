#!/usr/bin/env bash

SRCDIR=/local/wasiahmad/workspace/projects/PrivacyIE/unilm/data

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
for split in train valid test; do
    python format.py \
        --source $SRCDIR/$split.source \
        --target $SRCDIR/$split.target \
        --outfile $SRCDIR/$split.json;
done
