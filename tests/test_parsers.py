from tinygen import preprocess, train
from tinygen.tinygen import build_subparsers


def test_train_parser_arguments() -> None:
    parser = build_subparsers()
    args = parser.parse_args(
        [
            "train",
            "--train_dataset_path",
            "/path/to/train_dataset.csv",
            "--output_path",
            "/path/to/output_dir",
            "--num_classes",
            "3",
        ]
    )
    assert args.command == "train"
    pars = train.preprocess_parameters(vars(args))
    # mandatory
    assert pars.train_dataset_path == "/path/to/train_dataset.csv"
    assert pars.output_path == "/path/to/output_dir"
    assert pars.num_classes == 3
    # optional
    assert pars.eval_dataset_path is None
    assert pars.shuffle_buffer_size == 0
    assert pars.batch_size == 32
    assert pars.epochs == 10


def test_preprocess_parser_arguments() -> None:
    parser = build_subparsers()
    args = parser.parse_args(
        [
            "preprocess",
            "--input_file",
            "/path/to/train_dataset.csv",
            "--output_path",
            "/path/to/output_dir",
            "--class_mapping",
            '{"class1": 0, "class2": 1}',
        ]
    )

    assert args.command == "preprocess"
    pars = preprocess.preprocess_parameters(vars(args))
    # mandatory
    assert pars.input_file == "/path/to/train_dataset.csv"
    assert pars.output_path == "/path/to/output_dir"
    assert pars.class_mapping == {"class1": 0, "class2": 1}
