import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from pathlib import Path

from ludwig.results import gen_param_paths

from startingabstract import __name__
from startingabstract.figs import make_dp_unigram_fig
from startingabstract.params import param2default, param2requests

RESEARCH_DATA_PATH: Optional[Path] = Path('/media/research_data')
RUNS_PATH = None  # config.Dirs.runs  # config.Dirs.runs if using local results or None if using results form Ludwig
FILE_NAME: str = 'dp_all-verbs-4096_part0.csv'                   # contains trajectory of some performance measure

LABEL_N: bool = True                       # add information about number of replications to legend
PLOT_MAX_LINES: bool = False                # plot horizontal line at best overall performance
PLOT_MAX_LINE: bool = False                 # plot horizontal line at best performance for each param
PALETTE_IDS: Optional[List[int]] = [0, 1]   # re-assign colors to each line
V_LINES: Optional[List[int]] = None       # add vertical lines to highlight time slices
TITLE: str = ''
LABELS: Optional[List[str]] = ['reverse age-ordered', 'age-ordered']  # custom labels for figure legend
FIG_SIZE: Tuple[int, int] = (8, 6)  # in inches
Y_LIMS: List[float] = [0, 4]
Y_LABEL: str = 'Bits Divergence from Verb Prototype' or FILE_NAME


def make_summary(param_path, label):
    """
    load all csv files matching FILENAME and return mean and std across their contents
    """
    series_list = [pd.read_csv(p, index_col=0, squeeze=True) for p in param_path.rglob(FILE_NAME)]
    concatenated_df = pd.concat(series_list, axis=1)
    grouped = concatenated_df.groupby(by=concatenated_df.columns, axis=1)
    y_mean = grouped.mean().values.flatten()
    y_std = grouped.std().values.flatten()
    return concatenated_df.index, y_mean, y_std, label, len(series_list)


# collect summaries
summaries = []
param2requests['reverse'] = [False, True]  # do not show results for shuffled_docs=True
project_name = __name__
for p, label in gen_param_paths(project_name,
                                param2requests,
                                param2default,
                                runs_path=RUNS_PATH,
                                research_data_path=RESEARCH_DATA_PATH,
                                label_n=LABEL_N):
    summary = make_summary(p, label)  # summary contains: x, mean_y, std_y, label, n
    summaries.append(summary)
    print(f'--------------------- End section {p.name}')
    print()




# plot
fig = make_dp_unigram_fig()
fig.show()