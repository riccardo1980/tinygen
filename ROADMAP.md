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

## Next steps
- use gcs_to_fuse to path folders   
- scaffold train section
- preprocess:
    - create task driven preprocessors
    - rewrite
        - tfrecords.do_pipeline:
            - rewrite using preprocess.run code
            - rewrite tests/io/test_tfrecords.py::test_roundtrip
- test coverage:
    - create tests for all pipline steps called in preprocess.run


______
steps in preprocess.run
- read file
- filter for allowed classes
    - filter applies the filter: instead use filter for lambda and call the built-in filter functiion outside
- branches
    - stats
    - write
        - formatter: make_formatter
        - serializer: lambda
        - actual write: 