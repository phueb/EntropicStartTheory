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
BA_TYPE: str = 'ba_o'
PROBES_NAME: str = 'sem-4096'

LABEL_N: bool = True                       # add information about number of replications to legend
PLOT_MAX_LINES: bool = False                # plot horizontal line at best overall performance
PLOT_MAX_LINE: bool = False                 # plot horizontal line at best performance for each param
PALETTE_IDS: Optional[List[int]] = [1, 0]   # re-assign colors to each line
V_LINES: Optional[List[int]] = None       # add vertical lines to highlight time slices
LABELS: Optional[List[str]] = None # ['reverse age-ordered', 'age-ordered']  # custom labels for figure legend
FIG_SIZE: Tuple[int, int] = (8, 6)  # in inches
Y_LIMS: List[float] = [0.5, 0.8]
Y_LABEL: str = 'Balanced Accuracy'

param2requests['legacy'] = [True]


def correct_artifacts(y, tolerance=0.00):
    """
    correct y when y drops more than tolerance.
    this is necessary because computation of balanced accuracy occasionally results in unwanted negative spikes
    """
    res = np.asarray(y)
    for i in range(len(res) - 2):
        val1, val2, val3 = res[[i, i+1, i+2]]
        if (val1 - tolerance) > val2 < (val3 - tolerance):
            res[i+1] = np.mean([val1, val3])
            print('Adjusting {} to {}'.format(val2, np.mean([val1, val3])))
    return res.tolist()


def make_summary(param_path, label):
    """
    load all csv files matching FILENAME and return mean and std across their contents
    """
    file_name = f'{BA_TYPE}_{PROBES_NAME}.csv'
    series_list = [pd.read_csv(p, index_col=0, squeeze=True) for p in param_path.rglob(file_name)]
    concatenated_df = pd.concat(series_list, axis=1)
    grouped = concatenated_df.groupby(by=concatenated_df.columns, axis=1)
    y_mean = grouped.mean().values.flatten()
    y_std = grouped.std().values.flatten()
    return concatenated_df.index, y_mean, y_std, label, len(series_list)


# collect summaries
summaries = []
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

# sort data
summaries = sorted(summaries, key=lambda s: s[1][-1], reverse=True)
if not summaries:
    raise SystemExit('No data found')

# print to console
for s in summaries:
    _, y_mean, y_std, label, n = s
    print(label)
    print(y_mean)
    print(y_std)
    print()

# plot
fig = make_summary_fig(summaries,
                       Y_LABEL,
                       title=f'{BA_TYPE}_{PROBES_NAME}.csv',
                       palette_ids=PALETTE_IDS,
                       figsize=FIG_SIZE,
                       ylims=Y_LIMS,
                       legend_labels=LABELS,
                       vlines=V_LINES,
                       plot_max_lines=PLOT_MAX_LINES,
                       plot_max_line=PLOT_MAX_LINE,
                       legend_loc='best',
                       )
fig.show()