import pandas as pd
from typing import Optional, List, Tuple
from pathlib import Path

from ludwig.results import gen_param_paths

from startingabstract import __name__
from startingabstract.figs import make_summary_fig
from startingabstract.params import param2default, param2requests

RESEARCH_DATA_PATH: Optional[Path] = Path('/media/research_data')
RUNS_PATH = None  # config.Dirs.runs if using local results or None if using results form Ludwig
DP_PROBES_NAME: str = 'singular-nouns-4096'
PART_ID = 0
METRIC = 'xe'


Y_LABEL = 'Divergence from Prototype (Unconditional)'
LABEL_N: bool = True
FIG_SIZE: Tuple[int, int] = (8, 6)  # in inches
Y_LIMS: List[float] = [0, 13]
PARAMS_AS_TITLE: bool = True
LOG_X: bool = False

PLOT_SINGLE_SUMMARY = True

# param2requests['shuffle_sentences'] = [True]


def make_summary(pp, lb):
    """
    load all csv files for dp-unconditional analysis
    """
    pattern = f'dp_{DP_PROBES_NAME}_part{PART_ID}_{METRIC}_unconditional.csv'
    series_list = [pd.read_csv(p, index_col=0, squeeze=True) for p in pp.rglob(pattern)]
    if not series_list:
        raise RuntimeError(f'Did not find any csv files matching pattern="{pattern}"')
    concatenated_df = pd.concat(series_list, axis=1)
    grouped = concatenated_df.groupby(by=concatenated_df.columns, axis=1)
    y_mean = grouped.mean().values.flatten()
    y_std = grouped.std().values.flatten()
    return concatenated_df.index, y_mean, y_std, lb, len(series_list)


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
                       title=f'{DP_PROBES_NAME}\npartition={PART_ID}',
                       log_x=LOG_X,
                       ylims=Y_LIMS,
                       figsize=FIG_SIZE,
                       legend_loc='upper left',
                       # legend_labels=['reverse age-ordered', 'age-ordered'],
                       palette_ids=[0, 1],  # re-assign colors to each line
                       )
fig.show()