#!/usr/bin/env bash

set -e

SRCS="tinygen"

[ -d "$SRCS" ] || (echo "Run this script from project root"; exit 1)

TRAIN_DATASET="data/SMSSpamCollection/train/tfrecords"
EVAL_DATASET="data/SMSSpamCollection/train/tfrecords"
NUM_CLASSES=2
OUTPUT_PATH="models/01"
BATCH_SIZE=2
EPOCHS=10
DROPOUT=0.9

python -m tinygen.tinygen train \
    --train_dataset_path ${TRAIN_DATASET} \
    --eval_dataset_path ${EVAL_DATASET} \
    --num_classes ${NUM_CLASSES} \
    --output_path ${OUTPUT_PATH} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --dropout ${DROPOUT}
