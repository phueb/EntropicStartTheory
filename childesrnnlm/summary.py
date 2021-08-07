from pathlib import Path
import numpy as np
from typing import Tuple, Optional, Union, List
import yaml
import pandas as pd
from scipy.stats import sem, t

from childesrnnlm import configs
from childesrnnlm.params import Params


def make_summary(pattern: str,
                 param_path: Path,
                 label: str,
                 confidence: float,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, Union[int, None]]:
    """
    load all csv files matching pattern and return mean and std across their contents
    """
    pattern = f'{pattern}.csv'
    series_list = [pd.read_csv(p, index_col=0, squeeze=True)
                   for p in param_path.rglob(pattern)]
    n = len(series_list)
    if not series_list:
        raise RuntimeError(f'Did not find any csv files matching pattern="{pattern}"')
    concatenated_df = pd.concat(series_list, axis=1)
    x = concatenated_df.index.values
    y_mean = concatenated_df.mean(axis=1).values.flatten()
    y_sem = sem(concatenated_df.values, axis=1)
    h = y_sem * t.ppf((1 + confidence) / 2, n - 1)  # margin of error

    # load params to get info about how much to shift x due to entropic-start
    with (param_path / 'param2val.yaml').open('r') as f:
        param2val = yaml.load(f, Loader=yaml.FullLoader)
    params = Params.from_param2val(param2val)

    # align curves to start of training corpus, not start of artificial pre-training data  # todo improve
    if params.start != 'none':
        # the calculation is not adjusted for pruning performed by preppy
        num_probes = 699
        num_start_sequences = configs.Start.num_right_words * configs.Start.num_left_words * num_probes
        num_start_tokens = num_start_sequences * 3
        num_shifted_steps = num_start_tokens // params.batch_size * params.num_iterations[0]

        print(f'Shifting x axis by {num_shifted_steps}')
        x -= num_shifted_steps

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
