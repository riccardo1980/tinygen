FROM python:3.10.4-slim as build-stage

ENV POETRY_VERSION=1.7.1

RUN pip install "poetry==${POETRY_VERSION}"

WORKDIR /app

COPY poetry.lock pyproject.toml ./

RUN poetry export \
    --without-hashes --without dev \
    -f requirements.txt --output requirements.txt 

FROM python:3.10.4-slim

WORKDIR /app

COPY --from=build-stage /app/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY tinygen tinygen

ENV PYTHONPATH "."

ENTRYPOINT ["python", "-m", "tinygen.tinygen", "train"]
