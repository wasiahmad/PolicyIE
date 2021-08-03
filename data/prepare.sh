#!/bin/sh

out_dir=policyie

# ****************** DATA PREPARATION ****************** #

function prepare () {

unzip sanitized_split.zip

# pre-processing data from json to BIO tagging format
python process.py \
    --input_paths sanitized_split/train \
    --out_dir ${out_dir}/train;

python process.py \
    --input_paths sanitized_split/test \
    --out_dir ${out_dir}/test;

for split in train test; do
    python del_partial_slots.py --input_dir ${out_dir}/${split}
    python del_partial_slots.py --input_dir ${out_dir}/${split}
done

# Shorten labels
python shorten_labels.py --input_path ${out_dir}/train --out_dir ${out_dir}/train
python shorten_labels.py --input_path ${out_dir}/test --out_dir ${out_dir}/test

# sanity check for BIO tagging data
for split in train test; do
    python BIO_sanity_check.py --input_file ${out_dir}/${split}/seq_type_I.out
    python BIO_sanity_check.py --input_file ${out_dir}/${split}/seq_type_II.out
    python BIO_sanity_check.py --input_file ${out_dir}/${split}/short_seq_type_I.out
    python BIO_sanity_check.py --input_file ${out_dir}/${split}/short_seq_type_II.out
done

# convert the data into compositional form
python flattener.py --data_dir ${out_dir}
# generate label (intent, slots and postag) files
python labeler.py --data_dir ${out_dir}

rm -rf sanitized_split
rm -rf __MACOSX

}

prepare


# ****************** Seq2Seq DATA FORMATTING ****************** #

function s2s_data_formatting () {

dest_dir=s2s_format
mkdir -p $dest_dir

cp $out_dir/train/seq.in $dest_dir/temp.source
cp $out_dir/train/seq.out $dest_dir/temp.target
cp $out_dir/test/seq.in $dest_dir/test.source
cp $out_dir/test/seq.out $dest_dir/test.target

tail -100 $dest_dir/temp.source > $dest_dir/valid.source
tail -100 $dest_dir/temp.target > $dest_dir/valid.target
head -n -100 $dest_dir/temp.source > $dest_dir/train.source
head -n -100 $dest_dir/temp.target > $dest_dir/train.target
rm $dest_dir/temp.source && rm $dest_dir/temp.target

}

s2s_data_formatting


# ****************** BIO-tagging DATA FORMATTING ****************** #

function bio_formatting () {

dest_dir=bio_format
mkdir -p $dest_dir

mkdir -p $dest_dir
mkdir -p $dest_dir/train
mkdir -p $dest_dir/valid
mkdir -p $dest_dir/test

cp $out_dir/train/seq.in $dest_dir/temp.seq
cp $out_dir/train/seq_type_I.out $dest_dir/temp.seq_type_I
cp $out_dir/train/seq_type_II.out $dest_dir/temp.seq_type_II
cp $out_dir/train/label $dest_dir/temp.label
cp $out_dir/train/pos_tags.out $dest_dir/temp.pos_tags

# valid
tail -100 $dest_dir/temp.seq > $dest_dir/valid/seq.in
tail -100 $dest_dir/temp.seq_type_I > $dest_dir/valid/seq_type_I.out
tail -100 $dest_dir/temp.seq_type_II > $dest_dir/valid/seq_type_II.out
tail -100 $dest_dir/temp.label > $dest_dir/valid/label
tail -100 $dest_dir/temp.pos_tags > $dest_dir/valid/pos_tags.out

# train
head -n -100 $dest_dir/temp.seq > $dest_dir/train/seq.in
head -n -100 $dest_dir/temp.seq_type_I > $dest_dir/train/seq_type_I.out
head -n -100 $dest_dir/temp.seq_type_II > $dest_dir/train/seq_type_II.out
head -n -100 $dest_dir/temp.label > $dest_dir/train/label
head -n -100 $dest_dir/temp.pos_tags > $dest_dir/train/pos_tags.out

rm $dest_dir/temp.seq && rm $dest_dir/temp.label
rm $dest_dir/temp.seq_type_I && rm $dest_dir/temp.seq_type_II
rm $dest_dir/temp.pos_tags

# test files
cp $out_dir/test/seq.in $dest_dir/test/seq.in
cp $out_dir/test/seq_type_I.out $dest_dir/test/seq_type_I.out
cp $out_dir/test/seq_type_II.out $dest_dir/test/seq_type_II.out
cp $out_dir/test/label $dest_dir/test/label
cp $out_dir/test/pos_tags.out $dest_dir/test/pos_tags.out

# label files
cp $out_dir/type_II_slot_label.txt $dest_dir/type_II_slot_label.txt
cp $out_dir/type_I_slot_label.txt $dest_dir/type_I_slot_label.txt
cp $out_dir/intent_label.txt $dest_dir/intent_label.txt
cp $out_dir/postag_label.txt $dest_dir/postag_label.txt

python vocab.py \
--data_dir $dest_dir \
--vocab_file ${dest_dir}/vocab.txt;

}

bio_formatting


# ****************** BIO-tagging DATA FORMATTING ****************** #

function download () {
    FILE=polisis-300d-137M-subword.txt
    if [[ ! -f $FILE ]]; then
        fileid="1EIwu1ahCmoHkAIpnrG-fmosbu3qsCbda"
        baseurl="https://drive.google.com/uc?export=download"
        curl -c ./cookie -s -L "${baseurl}&id=${fileid}" > /dev/null
        curl -Lb ./cookie "${baseurl}&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${FILE}
        rm cookie
    fi
}

download