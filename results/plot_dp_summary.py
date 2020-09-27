import pandas as pd
from typing import Optional, List, Tuple
from pathlib import Path
import numpy as np
from scipy.stats import t, sem

from ludwig.results import gen_param_paths

from provident import __name__
from provident.figs import make_summary_fig
from provident.params import param2default, param2requests

RESEARCH_DATA_PATH: Optional[Path] = Path('/media/research_data')
RUNS_PATH = None  # config.Dirs.runs if using local results or None if using results form Ludwig
DP_PROBES_NAME: str = 'sem-4096'
METRIC = 'js'
PART_ID = 0

Y_LABEL = 'Divergence from Prototype\n +/- 95%-CI'
LABEL_N: bool = True
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
Y_LIMS: List[float] = [0.0, 0.8]
X_LIMS: Optional[List[int]] = None  # [0, 100_000]
LOG_X: bool = False
CONFIDENCE = 0.95
TITLE = ''  # f'{DP_PROBES_NAME}\npartition={PART_ID}'

param2requests['legacy'] = [True]


def make_summary(pp, lb) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, int]:
    """
    load all csv files for dp-unconditional results
    """
    pattern = f'dp_{DP_PROBES_NAME}_{METRIC}.csv'
    series_list = [pd.read_csv(p, index_col=0, squeeze=True) for p in pp.rglob(pattern)]
    n = len(series_list)
    if not series_list:
        raise RuntimeError(f'Did not find any csv files matching pattern="{pattern}"')
    concatenated_df = pd.concat(series_list, axis=1)
    y_mean = concatenated_df.mean(axis=1).values.flatten()
    y_sem = sem(concatenated_df.values, axis=1)
    h = y_sem * t.ppf((1 + CONFIDENCE) / 2, n - 1)  # margin of error

    return concatenated_df.index.values, y_mean, h, lb, n


# collect summaries
summaries = []
project_name = __name__
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         runs_path=RUNS_PATH,
                                         research_data_path=RESEARCH_DATA_PATH,
                                         label_n=LABEL_N):
    # collect for comparison figure
    summary = make_summary(param_path, label)
    summaries.append(summary)

# plot comparison
fig = make_summary_fig(summaries,
                       ylabel=Y_LABEL,
                       title=TITLE,
                       log_x=LOG_X,
                       ylims=Y_LIMS,
                       xlims=X_LIMS,
                       figsize=FIG_SIZE,
                       legend_loc='best',
                       vline=100_000,
                       # legend_labels=['reverse age-ordered', 'age-ordered'],
                       palette_ids=[0, 1],  # re-assign colors to each line
                       )
fig.show()