from typing import Optional, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from ludwig.results import gen_param_paths

from childesrnnlm import __name__
from childesrnnlm.figs import make_summary_fig
from childesrnnlm.params import param2default, param2requests

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH = None  # config.Dirs.runs if using local plot or None if using plot form Ludwig
PROBES_NAME: str = 'sem-2021'

LABEL_N: bool = True
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
CONFIDENCE: float = 0.95
TITLE = ''
PERFORMANCE_NAME = ['ra', 'ba', 'dp', 'ws', 'as', 'si', 'sd'][4]

if PERFORMANCE_NAME == 'ra':
    Y_LABEL = 'Raggedness of In-Out Mapping\n+/- 95%-CI'
    Y_LIMS: List[float] = [0, 30]
elif PERFORMANCE_NAME == 'ba':
    Y_LABEL = 'Balanced Accuracy\n+/- 95%-CI'
    Y_LIMS: List[float] = [0.5, 0.7]
elif PERFORMANCE_NAME == 'dp':
    Y_LABEL = 'Divergence from Prototype\n+/- 95%-CI'
    Y_LIMS: List[float] = [0.0, 1.0]
elif PERFORMANCE_NAME == 'ws':
    Y_LABEL = 'Within-Category Spread\n+/- 95%-CI'
    Y_LIMS: List[float] = [0.0, 13.0]
elif PERFORMANCE_NAME == 'as':
    Y_LABEL = 'Across-Category Spread\n+/- 95%-CI'
    Y_LIMS: List[float] = [200, 350]
elif PERFORMANCE_NAME == 'si':
    Y_LABEL = 'Silhouette Score\n+/- 95%-CI'
    Y_LIMS: List[float] = [-0.1, 0.0]
elif PERFORMANCE_NAME == 'sd':
    Y_LABEL = 'S_Dbw Score\n+/- 95%-CI'
    Y_LIMS: List[float] = [0.9, 1.0]
else:
    raise AttributeError

# collect summaries
summaries = []
project_name = __name__
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         runs_path=RUNS_PATH,
                                         ludwig_data_path=LUDWIG_DATA_PATH,
                                         label_n=LABEL_N):
    pattern = f'{PERFORMANCE_NAME}_n_{PROBES_NAME}.csv'
    for p in param_path.rglob(pattern):
        s = pd.read_csv(p, index_col=0, squeeze=True)
        n = 1
        y_mean = s.values
        h = np.zeros((len(s)))  # margin of error

        summary = (s.index.values, y_mean, h, label, n)
        summaries.append(summary)

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
                       ylabel=Y_LABEL,
                       title=TITLE,
                       ylims=Y_LIMS,
                       figsize=FIG_SIZE,
                       legend_loc='best',
                       annotate=True,
                       )
fig.show()
