from pathlib import Path
import numpy as np
from typing import Tuple, Optional, Union, List

import pandas as pd
from scipy.stats import sem, t


def make_summary(pattern: str,
                 path_to_search: Path,
                 label: str,
                 confidence: float,
                 shift_x: Optional[int] = None,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Union[int, None]]:
    """
    load all csv files matching pattern and return mean and std across their contents
    """
    pattern = f'{pattern}.csv'
    series_list = [pd.read_csv(p, index_col=0, squeeze=True)
                   for p in path_to_search.rglob(pattern)]
    n = len(series_list)
    if not series_list:
        raise RuntimeError(f'Did not find any csv files matching pattern="{pattern}"')
    concatenated_df = pd.concat(series_list, axis=1)
    x = concatenated_df.index.values
    y_mean = concatenated_df.mean(axis=1).values.flatten()
    y_sem = sem(concatenated_df.values, axis=1)
    h = y_sem * t.ppf((1 + confidence) / 2, n - 1)  # margin of error

    if shift_x is not None:
        print(f'Shifting x axis by {shift_x}')
        x -= shift_x

    job_id = None  # this is useful only when a summary corresponds to an individual job

    return x, y_mean, h, label, job_id


def sort_and_print_summaries(summaries: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, Union[int, None]]],
                             ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, Union[int, None]]]:
    # sort
    summaries = sorted(summaries, key=lambda s: s[1][-1], reverse=True)
    if not summaries:
        raise SystemExit('No data found')

    # print to console
    for s in summaries:
        _, y_mean, y_std, label, job_id = s
        print()
        print(label)
        print(job_id)
        print(np.round(y_mean, 2))

    return summaries
