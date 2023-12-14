from typing import Dict

import pandas as pd
import pytest

from tinygen.io.utils import extract_labels


@pytest.mark.parametrize(
    "input, expected",
    [
        (pd.Series(["ham", "ham", "spam", "spam"]), {"ham": 0, "spam": 1}),
        (
            pd.Series(["ham", "ham", "speck", "spam"]),
            {"ham": 0, "spam": 1, "speck": 2},
        ),
    ],
)
def test_extract_labels(input: pd.Series, expected: Dict) -> None:
    assert extract_labels(input) == expected
