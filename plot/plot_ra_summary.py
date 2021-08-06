from typing import Optional, List, Tuple
from pathlib import Path
import yaml

from ludwig.results import gen_param_paths

from childesrnnlm import __name__, configs
from childesrnnlm.figs import make_summary_fig
from childesrnnlm.summary import make_summary
from childesrnnlm.params import param2default, param2requests, Params

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH = None  # config.Dirs.runs  # config.Dirs.runs if using local plot or None if using plot form Ludwig
RA_TYPE: str = 'ra_n'
PROBES_NAME: str = 'sem-2021'

LABEL_N: bool = True                       # add information about number of replications to legend
PLOT_MAX_LINE: bool = False                 # plot horizontal line at best performance for each param
PLOT_MAX_LINES: bool = False                # plot horizontal line at best overall performance
PALETTE_IDS: Optional[List[int]] = None   # re-assign colors to each line
V_LINES: Optional[List[int]] = None       # add vertical lines to highlight time slices
LABELS: Optional[List[str]] = None  # ['reverse age-ordered', 'age-ordered']  # custom labels for figure legend
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
Y_LIMS: List[float] = [0.0, 18.0]
CONFIDENCE: float = 0.95
TITLE = ''  # f'{RA_TYPE}_{PROBES_NAME}.csv'

Y_LABEL: str = f'Raggedness of In-Out Mapping\n +/- 95%-CI'

# collect summaries
summaries = []
project_name = __name__
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         runs_path=RUNS_PATH,
                                         ludwig_data_path=LUDWIG_DATA_PATH,
                                         label_n=LABEL_N):

    # load params
    with (param_path / 'param2val.yaml').open('r') as f:
        param2val = yaml.load(f, Loader=yaml.FullLoader)
    params = Params.from_param2val(param2val)

    # TODO get num_shifted_steps automatically from information in entropicstart project rather than manually

    # align curves to start of training corpus, not start of artificial pre-training data
    if params.start != 'none':
        # the calculation is not adjusted for pruning performed by preppy
        num_probes = 699
        num_start_sequences = configs.Start.num_right_words * configs.Start.num_left_words * num_probes
        num_start_tokens = num_start_sequences * 3
        num_shifted_steps = num_start_tokens // params.batch_size * params.num_iterations[0]
    else:
        num_shifted_steps = None

    pattern = f'{RA_TYPE}_{PROBES_NAME}'
    summary = make_summary(pattern, param_path, label, CONFIDENCE, num_shifted_steps)
    summaries.append(summary)   # summary contains: x, mean_y, std_y, label, n
    print(f'--------------------- End section {param_path.name}')
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
                       title=TITLE,
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