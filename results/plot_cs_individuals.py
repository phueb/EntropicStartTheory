import pandas as pd
from typing import Optional, List, Tuple
from pathlib import Path
import numpy as np

from ludwig.results import gen_param_paths

from provident import __name__
from provident.figs import make_summary_fig
from provident.params import param2default, param2requests

RESEARCH_DATA_PATH: Optional[Path] = Path('/media/research_data')
RUNS_PATH = None  # config.Dirs.runs if using local results or None if using results form Ludwig
PROBES_NAME: str = 'sem-4096'
PART_ID = 0

Y_LABEL = 'Jensen-Shannon Divergence\nNoun vs. Noun'
LABEL_N: bool = True
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
Y_LIMS: List[float] = [0, 0.7]
PARAMS_AS_TITLE: bool = True
LOG_X: bool = False


summaries = []
project_name = __name__
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         runs_path=RUNS_PATH,
                                         research_data_path=RESEARCH_DATA_PATH,
                                         label_n=LABEL_N):
    pattern = f'cs_{PROBES_NAME}_js.csv'
    for p in param_path.rglob(pattern):
        s = pd.read_csv(p, index_col=0, squeeze=True)
        n = 1
        y_mean = s.values
        h = np.zeros((len(s)))  # margin of error

        # collect for comparison figure
        summary = (s.index.values, y_mean, h, label, n)

        summaries.append(summary)

fig = make_summary_fig(summaries,
                       ylabel=Y_LABEL,
                       title='',
                       log_x=LOG_X,
                       ylims=Y_LIMS,
                       figsize=FIG_SIZE,
                       legend_loc='best',
                       vline=200_000,
                       )
fig.show()