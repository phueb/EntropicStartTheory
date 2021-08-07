from typing import Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from ludwig.results import gen_param_paths

from childesrnnlm import __name__
from childesrnnlm.figs import make_summary_fig, get_y_label_and_lims
from childesrnnlm.summary import sort_and_print_summaries
from childesrnnlm.params import param2default, param2requests

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH = None  # config.Dirs.runs if using local plot or None if using plot form Ludwig
PROBES_NAME: str = 'sem-2021'

LABEL_N: bool = True
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
CONFIDENCE: float = 0.95
TITLE = ''
PERFORMANCE_NAME = ['ma',  # 0
                    'ra',  # 1
                    'ba',  # 2
                    'th',  # 3
                    'dp',  # 4
                    'du',  # 5
                    'ws',  # 6
                    'as',  # 7
                    'ed',  # 8
                    'cd',  # 9
                    'si',  # 10
                    'sd',  # 11
                    ][8]

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
y_label, y_lims = get_y_label_and_lims(PERFORMANCE_NAME)
fig = make_summary_fig(summaries,
                       ylabel=y_label,
                       title=TITLE,
                       ylims=y_lims,
                       figsize=FIG_SIZE,
                       legend_loc='best',
                       annotate=True,
                       )
fig.show()
