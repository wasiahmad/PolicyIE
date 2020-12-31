#!/bin/sh

unzip sanitized_split.zip

out_dir="non_overlapping_cross_sentence_annot"

# pre-processing data from json to BIO tagging format
python cross_sent_annot.py \
    --input_paths sanitized_split/train \
    --out_dir ${out_dir}/train;

python cross_sent_annot.py \
    --input_paths sanitized_split/test \
    --out_dir ${out_dir}/test;

# Shorten labels
python shorten_labels.py --input_path ${out_dir}/train --out_dir ${out_dir}/train
python shorten_labels.py --input_path ${out_dir}/test --out_dir ${out_dir}/test

# sanity check for BIO tagging data
for split in train test; do
    python BIO_sanity_check.py --input_file ${out_dir}/${split}/seq_entity.out
    python BIO_sanity_check.py --input_file ${out_dir}/${split}/seq_complex.out
    python BIO_sanity_check.py --input_file ${out_dir}/${split}/short_seq_entity.out
    python BIO_sanity_check.py --input_file ${out_dir}/${split}/short_seq_complex.out
done

# convert the data into compositional form
python flattener.py --data_dir ${out_dir}
# generate label (intent and slots) files
python labeler.py --data_dir ${out_dir}

rm -rf sanitized_split
rm -rf __MACOSX
