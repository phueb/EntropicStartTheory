from typing import Optional, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from ludwig.results import gen_param_paths

from childesrnnlm import __name__
from childesrnnlm.figs import make_summary_fig
from childesrnnlm.summary import sort_and_print_summaries
from childesrnnlm.params import param2default, param2requests

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH = None  # config.Dirs.runs if using local plot or None if using plot form Ludwig
PROBES_NAME: str = 'sem-2021'

LABEL_N: bool = True
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
CONFIDENCE: float = 0.95
TITLE = ''
PERFORMANCE_NAME = ['ma', 'ra', 'ba', 'dp', 'du', 'ws', 'as', 'si', 'sd'][8]

if PERFORMANCE_NAME == 'ma':
    Y_LABEL = 'Vector Magnitude\n+/- 95%-CI'
    Y_LIMS: List[float] = [0.5, 1.5]
elif PERFORMANCE_NAME == 'ra':
    Y_LABEL = 'Raggedness of In-Out Mapping\n+/- 95%-CI'
    Y_LIMS: List[float] = [0, 1]
elif PERFORMANCE_NAME == 'ba':
    Y_LABEL = 'Balanced Accuracy\n+/- 95%-CI'
    Y_LIMS: List[float] = [0.5, 0.7]
elif PERFORMANCE_NAME == 'dp':
    Y_LABEL = 'Divergence from Prototype\n+/- 95%-CI'
    Y_LIMS: List[float] = [0.0, 1.0]
elif PERFORMANCE_NAME == 'du':
    Y_LABEL = 'Divergence from Unigram Prototype\n+/- 95%-CI'
    Y_LIMS: List[float] = [0.0, 0.7]
elif PERFORMANCE_NAME == 'ws':
    Y_LABEL = 'Within-Category Spread\n+/- 95%-CI'
    Y_LIMS: List[float] = [0.0, 1.0]
elif PERFORMANCE_NAME == 'as':
    Y_LABEL = 'Across-Category Spread\n+/- 95%-CI'
    Y_LIMS: List[float] = [0, 16]
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
    for job_id, path_to_csv in enumerate(sorted(param_path.rglob(pattern))):
        s = pd.read_csv(path_to_csv, index_col=0, squeeze=True)
        y_mean = s.values
        h = np.zeros((len(s)))  # margin of error
        summary = (s.index.values, y_mean, h, label, job_id)
        summaries.append(summary)

summaries = sort_and_print_summaries(summaries)

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
