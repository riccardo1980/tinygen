# ROADMAP

## Features
- features:
    - text classification
    - text generation (completion)
- code management
    - :white_check_mark: testing
    - :white_check_mark: linting
- package
  - :white_check_mark: using poetry
  - docker image
- training
    - bare metal
    - docker
    - Vertex AI
- inference
    - inference package
    - inference API service
        - FastAPI
- dataset management
    - :white_check_mark: CSV to TFRecords

## Backlog
Use semantic classification: `<status>[module][class]<[branch]>`.  
### Classes
- TEST: add/rewrite tests
- DOC: improve documentation
- BUGFIX: fix a bug, potential bug
- FEATURE: add a feature
- CLEANCODE: improve code clarity

### Statuses
  - no status means not in development
  - :construction: `:construction:` add branch name 
  - :white_check_mark: `:white_check_mark:` means done and merged in devel 
  - no line means merged in master

Use [conventionalcommits](https://www.conventionalcommits.org/en/v1.0.0/)

### High priority
#### :white_check_mark: [TRAIN][FEATURE] uniform disk access in remote and local training
#### :white_check_mark: [TRAIN][FEATURE] create slim train image

#### [TRAIN][FEATURE] enable logging, model export
    - :white_check_mark: logging, checkpointing
    - :construction: save model
    - load model
    - add inference command
    - questions:
        - do I need do implement from_config()? [doc](https://keras.io/guides/serialization_and_saving/#model-serialization)

#### [TRAIN][FEATURE] remote GCP train
- Vertex Ai Custom training

#### [TRAIN][FEATURE] check for sharded input files

### Medium Priority
#### [ALL][CLEANCODE] define subset of conventional commit, update README.md accordingly

#### [PREPROCESS][FEATURE] allow for task driven preprocessors

#### [PREPROCESS][TEST] rewrite tests/io/test_tfrecords.py::test_roundtrip

#### [PREPROCESS][CLEANCODE] filter applies the filter: instead use filter for lambda and call the built-in filter function outside

#### [PREPROCESS][TEST] enhance coverage by defining tests for all pipline steps called in preprocess.run
- file read from CSV to iterable of records
- filter
- branches
  - stats
  - write
    - do_pipeline: formatting, tf.examples, serialization
    - write to tfrecords

## Low Priority
##### [PREPROCESS][BUGFIX] prevent unusable definition of label_to_index
  - check for indices fro m 0 to number_of_classes
  - where? used in:
    - make_formatter
    - others (?)

#####  [ALL][BUGFIX] Enforce typechecking by defining NamedTuple derived classes

#####  [PREPROCESS][BUGFIX] make_formatter: add input checks (pinned by a FIXME)
