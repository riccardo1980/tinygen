from typing import Dict

import pandas as pd


@DeprecationWarning
def extract_labels(series: pd.Series) -> Dict[str, int]:
    """
    Extract labels from dataframe

    :param df: dataframe
    :type df: pd.DataFrame
    :return: labels
        {"ham": 0, "spam": 1}
    :rtype: Dict[int, str]
    """

    labels = sorted(series.unique())
    label_to_index = {lbl: idx for idx, lbl in enumerate(labels)}
    return label_to_index
