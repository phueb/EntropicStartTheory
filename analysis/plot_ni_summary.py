import pandas as pd
from typing import Optional, List, Tuple
from pathlib import Path
from scipy.stats import t, sem

from ludwig.results import gen_param_paths

from startingabstract import __name__
from startingabstract.figs import make_summary_fig
from startingabstract.params import param2default, param2requests

RESEARCH_DATA_PATH: Optional[Path] = Path('/media/research_data')
RUNS_PATH = None  # config.Dirs.runs if using local results or None if using results form Ludwig
PROBES_NAME: str = 'syn-nva'
PART_ID = 0

Y_LABEL = 'Category-Separation\n+/- 95%-CI'
LABEL_N: bool = True
FIG_SIZE: Tuple[int, int] = (8, 6)  # in inches
Y_LIMS: List[float] = [0, 1]
PARAMS_AS_TITLE: bool = True
LOG_X: bool = False
CONFIDENCE: float = 0.95

PLOT_SINGLE_SUMMARY = False

param2requests['legacy'] = [True]


def make_summary(pp: Path, lb: str, pattern: str):
    """
    load all csv files matching pattern and return mean and ci across their contents
    """
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
summaries1 = []
summaries2 = []
summaries3 = []
project_name = __name__
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         runs_path=RUNS_PATH,
                                         research_data_path=RESEARCH_DATA_PATH,
                                         label_n=LABEL_N):
    # summary contains: x, mean_y, std_y, label, n
    summary1 = make_summary(param_path, label, f'ni_{PROBES_NAME}_NOUN_ADJ_js.csv')
    summary2 = make_summary(param_path, label, f'ni_{PROBES_NAME}_NOUN_NOUN_js.csv')
    summary3 = make_summary(param_path, label, f'ni_{PROBES_NAME}_NOUN_VERB_js.csv')

    # plot
    fig = make_summary_fig(summaries=[summary1, summary2, summary3],  # mus be in this order to match labels
                           ylabel=Y_LABEL,
                           title=f'{PROBES_NAME}\n{label}'if PARAMS_AS_TITLE else '',
                           log_x=LOG_X,
                           ylims=Y_LIMS,
                           figsize=FIG_SIZE,
                           legend_labels=['NOUN_ADJ', 'NOUN_NOUN', 'NOUN_VERB']
                           )
    if PLOT_SINGLE_SUMMARY:
        fig.show()

    # collect for comparison figure
    summaries1.append(summary1)
    summaries2.append(summary2)
    summaries3.append(summary3)

if PLOT_SINGLE_SUMMARY:
    raise SystemExit

# plot comparison
fig = make_summary_fig(summaries1,
                       ylabel=Y_LABEL,
                       title='NOUN_ADJ',
                       log_x=LOG_X,
                       ylims=Y_LIMS,
                       figsize=FIG_SIZE,
                       legend_loc='best',
                       vline=100_000,
                       # legend_labels=['reverse age-ordered', 'age-ordered'],
                       palette_ids=[0, 1],  # re-assign colors to each line
                       )
fig.show()

fig = make_summary_fig(summaries2,
                       ylabel=Y_LABEL,
                       title='NOUN_NOUN',
                       log_x=LOG_X,
                       ylims=Y_LIMS,
                       figsize=FIG_SIZE,
                       legend_loc='best',
                       vline=100_000,
                       # legend_labels=['reverse age-ordered', 'age-ordered'],
                       palette_ids=[0, 1],  # re-assign colors to each line
                       )
fig.show()

fig = make_summary_fig(summaries3,
                       ylabel=Y_LABEL,
                       title='NOUN_VERB',
                       log_x=LOG_X,
                       ylims=Y_LIMS,
                       figsize=FIG_SIZE,
                       legend_loc='best',
                       vline=100_000,
                       # legend_labels=['reverse age-ordered', 'age-ordered'],
                       palette_ids=[0, 1],  # re-assign colors to each line
                       )
fig.show()