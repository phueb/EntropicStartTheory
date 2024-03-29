from typing import Optional, List, Tuple
from pathlib import Path


from ludwig.results import gen_param_paths

from entropicstarttheory import __name__
from entropicstarttheory.figs import make_summary_fig
from entropicstarttheory.params import param2default, param2requests
from entropicstarttheory.summary import make_summary, sort_and_print_summaries

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH = None  # configs.Dirs.runs  # configs.Dirs.runs if loading runs locally or None if loading data from ludwig
WHICH_PP = ['train', 'test'][1]

LABEL_N: bool = True                       # add information about number of replications to legend
PLOT_MAX_LINES: bool = False                # plot horizontal line at best overall performance
PLOT_MAX_LINE: bool = False                 # plot horizontal line at best performance for each param
PALETTE_IDS: Optional[List[int]] = None   # re-assign colors to each line
V_LINES: Optional[List[int]] = [1_000_000]       # add vertical lines to highlight time slices
LABELS: Optional[List[str]] = None  # ['reverse age-ordered', 'age-ordered']  # custom labels for figure legend
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
Y_LIMS: List[float] = [10, 100]
CONFIDENCE: float = 0.95
TITLE = ''

if WHICH_PP == 'test':
    Y_LABEL: str = f'Test Perplexity \n +/- 95%-CI'
else:
    Y_LABEL: str = f'Train Perplexity \n +/- 95%-CI'

# collect summaries
summaries = []
project_name = __name__
for p, label in gen_param_paths(project_name,
                                param2requests,
                                param2default,
                                runs_path=RUNS_PATH,
                                ludwig_data_path=LUDWIG_DATA_PATH,
                                label_n=LABEL_N,
                                require_all_found=False,
                                ):
    pattern = f'{WHICH_PP}_pp'
    summary = make_summary(pattern, p, label, CONFIDENCE)  # summary contains: x, mean_y, margin-of-error, label, job_id
    summaries.append(summary)
    print(f'--------------------- End section {p.name}')
    print()

summaries = sort_and_print_summaries(summaries)

# plot
fig = make_summary_fig(summaries,
                       Y_LABEL,
                       log_y=True,
                       title=TITLE,
                       palette_ids=PALETTE_IDS,
                       figsize=FIG_SIZE,
                       ylims=Y_LIMS,
                       legend_labels=LABELS,
                       vlines=V_LINES,
                       plot_max_lines=PLOT_MAX_LINES,
                       plot_max_line=PLOT_MAX_LINE,
                       verbose=True,
                       )
fig.show()