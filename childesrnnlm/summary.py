from pathlib import Path
import numpy as np
from typing import Tuple, Optional, Union, List
import yaml
import pandas as pd
from scipy.stats import sem, t

from childesrnnlm import configs
from childesrnnlm.params import Params


def save_summary_to_txt(summary: Tuple[np.ndarray, np.ndarray, np.ndarray, str, Union[int, None]],
                        performance_name: str,
                        ) -> None:
    """
    output summary to text file.
    useful when plotting data with pgfplots on overleaf.org.

    notes:
        1. latex does not like spaces in file name
    """

    x, y_mean, h, label, job_id = summary

    # make path
    fn = performance_name + '_'
    fn += label.replace("\n", "-").replace(' ', '')
    path = configs.Dirs.summaries / f'{fn}.txt'  # txt format makes file content visible on overleaf.org
    if not path.parent.exists():
        path.parent.mkdir()

    # save to text
    df = pd.DataFrame(data={'mean': y_mean, 'margin_of_error': h}, index=list(x))
    df.index.name = 'step'
    df.round(3).to_csv(path, sep=' ')

    print(f'Saved summary to {path}')


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

    if not summaries:
        raise RuntimeError('No summaries were collected')

    # sort
    summaries = sorted(summaries, key=lambda s: s[1][-1], reverse=True)

    # print to console
    for s in summaries:
        _, y_mean, y_std, label, job_id = s
        print()
        print(label)
        print(job_id)
        print(np.round(y_mean, 2))

    return summaries
