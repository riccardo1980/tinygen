#!/usr/bin/env bash

set -e

SRCS="tinygen"

[ -d "$SRCS" ] || (echo "Run this script from project root"; exit 1)

# these folders will be made available to container
DATA_FOLDER='data/SMSSpamCollection_tiny/train/'
MODELS_FOLDER='models'

# TRAINER_PARAMETERS
N_FEATURES=128
NUM_CLASSES=3
NUM_CLASSES=2
BATCH_SIZE=2
EPOCHS=10
DROPOUT=0.9

###################################################################
PROJECT_ID=$(gcloud config get core/project)
CLOUD_ARTIFACT_ENDPOINT='europe-west1-docker.pkg.dev'
CLOUD_ARTIFACT_REPOSITORY='tinygen'
CLOUD_ARTIFACT_IMAGE='tinygen-trainer'
TAG='latest'
###################################################################
DATE=$(date +"%Y%m%d_%H%M%S")
JOBID="custom_model_${DATE}"
BASE_OUTPUT_DIRECTORY="$MODELS_FOLDER/$JOBID"
###################################################################

mkdir -p $BASE_OUTPUT_DIRECTORY

# paths available in the container
DATA_FOLDER_FROM_DOCKER="/data"
BASE_OUTPUT_DIRECTORY_FROM_DOCKER="/base_output_directory"
TRAIN_DATASET="$DATA_FOLDER_FROM_DOCKER/tfrecords"
EVAL_DATASET="$DATA_FOLDER_FROM_DOCKER/tfrecords"
MODEL_OUTPUT="$BASE_OUTPUT_DIRECTORY_FROM_DOCKER"

# parameters


#  trainer parameters
trainer_pars="
    --train_dataset_path ${TRAIN_DATASET} \
    --eval_dataset_path ${EVAL_DATASET} \
    --num_classes ${NUM_CLASSES} \
    --output_path ${BASE_OUTPUT_DIRECTORY} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --dropout ${DROPOUT}"
                
echo -e "\nlocal run through docker\n"

docker run \
    --rm \
    --mount type=bind,src=${PWD}/${BASE_OUTPUT_DIRECTORY},dst=${BASE_OUTPUT_DIRECTORY_FROM_DOCKER} \
    --mount type=bind,src=${PWD}/${DATA_FOLDER},dst=${DATA_FOLDER_FROM_DOCKER} \
    ${CLOUD_ARTIFACT_ENDPOINT}/${PROJECT_ID}/${CLOUD_ARTIFACT_REPOSITORY}/${CLOUD_ARTIFACT_IMAGE}:${TAG} \
    $trainer_pars

ret=$?
if [ $ret -eq 0 ]; then
    echo -e "\nSUCCESS\n"
else
    echo "error on local run"
fi
exit $ret