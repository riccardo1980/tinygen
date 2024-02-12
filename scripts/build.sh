#!/usr/bin/env bash

set -e

SRCS="tinygen"

[ -d "$SRCS" ] || (echo "Run this script from project root"; exit 1)

PROJECT_ID=$(gcloud config get core/project)
CLOUD_ARTIFACT_ENDPOINT='europe-west1-docker.pkg.dev'
CLOUD_ARTIFACT_REPOSITORY='tinygen'
CLOUD_ARTIFACT_IMAGE='tinygen-trainer'

TAG='latest'
DOCKERFILE='ci/trainer/Dockerfile'

docker build \
    -t ${CLOUD_ARTIFACT_ENDPOINT}/${PROJECT_ID}/${CLOUD_ARTIFACT_REPOSITORY}/${CLOUD_ARTIFACT_IMAGE}:${TAG}\
    -f ${DOCKERFILE} .