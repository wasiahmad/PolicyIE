#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8;
CURRENT_DIR=`pwd`
HOME_DIR=`realpath ..`

DATADIR=${HOME_DIR}/data/s2s_format
DESTDIR=${CURRENT_DIR}/resources
mkdir -p $DESTDIR

for split in train valid test; do
    python format.py \
        --source $DATADIR/$split.source \
        --target $DATADIR/$split.target \
        --outfile $DESTDIR/$split.json;
done
