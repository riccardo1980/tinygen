#!/usr/bin/env bash
set -e

SRCS="tinygen"

[ -d $SRCS ] || (echo "Run this script from project root"; exit 1)


poetry run coverage run -m pytest
poetry run coverage report --show-missing
poetry run coverage xml