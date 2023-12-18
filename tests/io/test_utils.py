import pytest

from tinygen.io.utils import convert_to_fuse


@pytest.mark.parametrize(
    "input_path, expected_output",
    [
        ("gs://my_bucket/aa/bb", "/gcs/my_bucket/aa/bb"),
        ("/path/to/local/file", "/path/to/local/file"),
    ],
)
def test_convert_to_fuse(input_path: str, expected_output: str) -> None:
    actual_output = convert_to_fuse(input_path)
    assert actual_output == expected_output
