from typing import Any, Dict, List

import pytest

from tinygen import preprocess, train_pars
from tinygen.tinygen import build_subparsers


@pytest.mark.parametrize(
    "input,expected",
    [
        (
            [
                "train",
                "--train_dataset_path",
                "/path/to/train_dataset.csv",
                "--output_path",
                "/path/to/output_dir",
                "--num_classes",
                "3",
            ],
            {
                "command": "train",
                "train_dataset_path": "/path/to/train_dataset.csv",
                "model_path": "/path/to/output_dir/model",
                "logs_path": "/path/to/output_dir/logs",
                "checkpoints_path": "/path/to/output_dir/checkpoints",
                "num_classes": 3,
            },
        ),
        (
            [
                "train",
                "--train_dataset_path",
                "gs://my-bucket/path/to/train_dataset.csv",
                "--output_path",
                "/path/to/output_dir",
                "--num_classes",
                "5",
                "--shuffle_buffer_size",
                "8",
                "--batch_size",
                "27",
                "--epochs",
                "13",
                "--eval_dataset_path",
                "gs://my-bucket/path/to/eval_dataset.csv",
                "--embedding_dim",
                "129",
                "--learning_rate",
                "0.1",
                "--dropout",
                "0.3",
            ],
            {
                "command": "train",
                "train_dataset_path": "/gcs/my-bucket/path/to/train_dataset.csv",
                "model_path": "/path/to/output_dir/model",
                "logs_path": "/path/to/output_dir/logs",
                "checkpoints_path": "/path/to/output_dir/checkpoints",
                "num_classes": 5,
                "eval_dataset_path": "/gcs/my-bucket/path/to/eval_dataset.csv",
                "shuffle_buffer_size": 8,
                "batch_size": 27,
                "epochs": 13,
                "embedding_dim": 129,
                "learning_rate": 0.1,
                "dropout": 0.3,
            },
        ),
    ],
)
def test_train_parser_arguments(input: List[str], expected: Dict[str, Any]) -> None:
    parser = build_subparsers()
    args = parser.parse_args(input)
    assert args.command == expected["command"]
    pars = train_pars.get_parameters(vars(args))
    # mandatory
    assert pars.train_dataset_path == expected["train_dataset_path"]
    assert pars.model_path == expected["model_path"]
    assert pars.logs_path == expected["logs_path"]
    assert pars.checkpoints_path == expected["checkpoints_path"]
    assert pars.num_classes == expected["num_classes"]
    # optional
    assert pars.eval_dataset_path == expected.get("eval_dataset_path", None)
    assert pars.shuffle_buffer_size == expected.get("shuffle_buffer_size", 0)
    assert pars.batch_size == expected.get("batch_size", 32)
    assert pars.epochs == expected.get("epochs", 10)
    assert pars.embedding_dim == expected.get("embedding_dim", 128)
    assert pars.learning_rate == expected.get("learning_rate", 0.001)
    assert pars.dropout == expected.get("dropout", 0.2)


@pytest.mark.parametrize(
    "input",
    [
        (
            [
                "train",
                "--train_dataset_path",
                "/path/to/train_dataset.csv",
                "--num_classes",
                "3",
            ]
        ),
    ],
)
def test_train_parser_arguments_raise_exception(input: List[str]) -> None:
    parser = build_subparsers()
    args = parser.parse_args(input)
    with pytest.raises(AssertionError, match="[^ ]* is not set"):
        train_pars.get_parameters(vars(args))


@pytest.mark.parametrize(
    "input,expected",
    [
        (
            [
                "preprocess",
                "--input_file",
                "/path/to/train_dataset.csv",
                "--output_path",
                "/path/to/output_dir",
                "--class_mapping",
                '{"class1": 0, "class2": 1}',
            ],
            {
                "command": "preprocess",
                "output_path": "/path/to/output_dir",
                "input_file": "/path/to/train_dataset.csv",
                "class_mapping": {"class1": 0, "class2": 1},
            },
        )
    ],
)
def test_preprocess_parser_arguments(
    input: List[str], expected: Dict[str, Any]
) -> None:
    parser = build_subparsers()
    args = parser.parse_args(input)

    assert args.command == expected["command"]
    pars = preprocess.preprocess_parameters(vars(args))

    assert pars.output_path == expected["output_path"]
    assert pars.input_file == expected["input_file"]
    assert pars.class_mapping == expected["class_mapping"]
