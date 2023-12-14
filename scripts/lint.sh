#!/usr/bin/env bash
set -e

SRCS="tinygen"
TEST_SRCS="tests"

[ -d "$SRCS" ] || (echo "Run this script from project root"; exit 1)


poetry run black "$SRCS" "$TEST_SRCS"
poetry run isort "$SRCS" "$TEST_SRCS"
poetry run mypy "$SRCS" "$TEST_SRCS"
poetry run flake8 "$SRCS" "$TEST_SRCS"