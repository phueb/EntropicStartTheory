import pandas as pd
from typing import Optional, List, Tuple
from pathlib import Path
import numpy as np

from ludwig.results import gen_param_paths

from entropicstarttheory import __name__
from entropicstarttheory.figs import make_summary_fig
from entropicstarttheory.params import param2default, param2requests

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH = None  # config.Dirs.runs if using local plot or None if using plot form Ludwig
DP_PROBES_NAME: str = 'sem-2021'
METRIC = 'js'
PART_ID = 0

Y_LABEL = 'Jensen-Shannon Divergence\nNoun vs. Noun-Prototype'
LABEL_N: bool = True
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
Y_LIMS: List[float] = [0.3, 0.8]
X_LIMS: Optional[List[int]] = None  # [0, 100_000]
LOG_Y: bool = False
CONFIDENCE = 0.95
TITLE = ''  # f'{DP_PROBES_NAME}\npartition={PART_ID}'


# collect summaries
summaries = []
project_name = __name__
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         runs_path=RUNS_PATH,
                                         ludwig_data_path=LUDWIG_DATA_PATH,
                                         label_n=LABEL_N):

    pattern = f'dp_{DP_PROBES_NAME}_{METRIC}.csv'
    for p in param_path.rglob(pattern):
        s = pd.read_csv(p, index_col=0, squeeze=True)
        n = 1
        y_mean = s.values
        h = np.zeros((len(s)))  # margin of error

        # collect for comparison figure
        summary = (s.index.values, y_mean, h, label, n)
        summaries.append(summary)

    if not summaries:
        raise RuntimeError(f'Did not find csv files matching {pattern}')

# plot comparison
fig = make_summary_fig(summaries,
                       ylabel=Y_LABEL,
                       title=TITLE,
                       log_y=LOG_Y,
                       ylims=Y_LIMS,
                       xlims=X_LIMS,
                       figsize=FIG_SIZE,
                       legend_loc='best',
                       # vline=200_000,
                       # legend_labels=['reverse age-ordered', 'age-ordered'],
                       # palette_ids=[0, 1],  # re-assign colors to each line
                       )
fig.show()