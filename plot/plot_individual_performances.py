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
RUNS_PATH = None  # configs.Dirs.runs if loading runs locally or None if loading data from ludwig

LABEL_N: bool = True
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
CONFIDENCE: float = 0.95
TITLE = ''

STRUCTURE_NAME: str = 'sem-2021'

DIRECTION = ['l',  # left-of-probe,
             'c',  # center (probe)
             'r',  # right-of-probe
             ][1]

LOCATION = ['inp',  # input layer
            'out',  # output layer
            ][0]

CONTEXT_TYPE = ['n',  # no context
                'o',  # ordered context
                ][1]

PERFORMANCE_NAME = ['ba',  # 0
                    'si',  # 1
                    'sd',  # 2
                    'ma',  # 3
                    'pr1',  # 4
                    'pr2',  # 5
                    'pd',  # 6
                    'pe',  # 7
                    'cs',  # 8
                    'cc',  # 9
                    'op',  # 10
                    'en',  # 11
                    'eo',  # 12
                    'fr',  # 13
                    ][0]


pattern = f'{PERFORMANCE_NAME}_{STRUCTURE_NAME}_{DIRECTION}_{LOCATION}_{CONTEXT_TYPE}.csv'


# collect summaries
summaries = []
project_name = __name__
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         runs_path=RUNS_PATH,
                                         ludwig_data_path=LUDWIG_DATA_PATH,
                                         label_n=LABEL_N):

    for job_id, path_to_csv in enumerate(sorted(param_path.rglob(pattern))):
        s = pd.read_csv(path_to_csv, index_col=0, squeeze=True)
        y_mean = s.values
        h = np.zeros((len(s)))  # margin of error
        summary = (s.index.values, y_mean, h, label, job_id)
        summaries.append(summary)

summaries = sort_and_print_summaries(summaries)

# plot
y_label, y_lims = get_y_label_and_lims(PERFORMANCE_NAME,
                                       DIRECTION,
                                       LOCATION,
                                       CONTEXT_TYPE,
                                       add_confidence_interval_to_label=False)
fig = make_summary_fig(summaries,
                       ylabel=y_label,
                       title=TITLE,
                       ylims=y_lims,
                       figsize=FIG_SIZE,
                       annotate=True,
                       )
fig.show()
