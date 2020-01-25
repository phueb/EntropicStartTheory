import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from pathlib import Path

from ludwig.results import gen_param_paths

from startingabstract import __name__
from startingabstract.figs import make_summary_fig
from startingabstract.params import param2default, param2requests

RESEARCH_DATA_PATH: Optional[Path] = Path('/media/research_data')
RUNS_PATH = None  # config.Dirs.runs  # config.Dirs.runs if using local results or None if using results form Ludwig
DP_NAME: str = 'singular-nouns-4096'

LABEL_N: bool = True                       # add information about number of replications to legend
PLOT_MAX_LINES: bool = False                # plot horizontal line at best overall performance
PLOT_MAX_LINE: bool = False                 # plot horizontal line at best performance for each param
PALETTE_IDS: Optional[List[int]] = [0, 1]   # re-assign colors to each line
V_LINES: Optional[List[int]] = None       # add vertical lines to highlight time slices
TITLE: str = ''
LABELS: Optional[List[str]] = ['model-based word', 'ideal word', 'ideal category']  # custom labels for figure legend
FIG_SIZE: Tuple[int, int] = (8, 6)  # in inches
Y_LIMS: List[float] = [0, 4]
Y_LABEL: str = 'Bits Divergence from Verb Prototype' or DP_NAME


def make_summary(pp, lb, pattern):
    """
    load all csv files for dp-unigram analysis
    """
    series_list = [pd.read_csv(p, index_col=0, squeeze=True) for p in pp.rglob(pattern)]
    concatenated_df = pd.concat(series_list, axis=1)

    print(concatenated_df)

    grouped = concatenated_df.groupby(by=concatenated_df.columns, axis=1)
    y_mean = grouped.mean().values.flatten()
    y_std = grouped.std().values.flatten()
    return concatenated_df.index, y_mean, y_std, lb, len(series_list)


# collect summaries
summaries = []
param2requests['reverse'] = [False, True]  # do not show results for shuffled_docs=True
project_name = __name__
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         runs_path=RUNS_PATH,
                                         research_data_path=RESEARCH_DATA_PATH,
                                         label_n=LABEL_N):
    # summary contains: x, mean_y, std_y, label, n
    summary1 = make_summary(param_path, label, f'dp_{DP_NAME}_unigram_1.csv')
    summary2 = make_summary(param_path, label, f'dp_{DP_NAME}_unigram_2.csv')
    summary3 = make_summary(param_path, label, f'dp_{DP_NAME}_unigram_3.csv')
    summaries = [summary1, summary2, summary3]  # mus be in this order to match labels

    # plot
    fig = make_summary_fig(summaries,
                           'Distance to Unigram Prototype',
                           title=label,
                           ylims=Y_LIMS,
                           alternative_labels=LABELS)
    fig.show()