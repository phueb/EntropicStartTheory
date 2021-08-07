from typing import Optional, List, Tuple
from pathlib import Path

from ludwig.results import gen_param_paths

from childesrnnlm import __name__
from childesrnnlm.figs import make_summary_fig
from childesrnnlm.summary import make_summary
from childesrnnlm.params import param2default, param2requests

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH = None  # config.Dirs.runs if using local plot or None if using plot form Ludwig
PROBES_NAME: str = 'sem-2021'

LABEL_N: bool = True
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
CONFIDENCE: float = 0.95
TITLE = ''
CS_TYPE = ['ws', 'as'][1]  # within or across

if CS_TYPE == 'ws':
    Y_LABEL = 'Within-Category Spread\n+/- 95%-CI'
    Y_LIMS: List[float] = [0.0, 13.0]
elif CS_TYPE == 'as':
    Y_LABEL = 'Across-Category Spread\n+/- 95%-CI'
    Y_LIMS: List[float] = [200, 350]
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

    num_shifted_steps = None  # TODO automate

    pattern = f'{CS_TYPE}_n_{PROBES_NAME}'
    summary = make_summary(pattern, param_path, label, CONFIDENCE, num_shifted_steps)
    summaries.append(summary)  # summary contains: x, mean_y, std_y, label, job_id
    print(f'--------------------- End section {param_path.name}')
    print()

# sort data
summaries = sorted(summaries, key=lambda s: s[1][-1], reverse=True)
if not summaries:
    raise SystemExit('No data found')

# print to console
for s in summaries:
    _, y_mean, y_std, label, job_id = s
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
                       )
fig.show()
