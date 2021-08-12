"""
Animation comparing average singular value plots between multiple conditions
"""
import numpy as np
from typing import Optional
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict
from celluloid import Camera

from ludwig.results import gen_param_paths

from childesrnnlm import __name__, configs
from childesrnnlm.params import param2default, param2requests


LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH: Optional[Path] = None  # configs.Dirs.runs if loading runs locally or None if from ludwig

WHERE = ['input', 'output'][0]
NUM_SINGULAR_DIMS = 16

if WHERE == 'output':
    Y_LIMS = [0, 0.5]
else:
    Y_LIMS = [0, 0.025]

param2requests = {'reverse': [True, False]}

# init collection
step2param_name2probe_reps_list = defaultdict(dict)  # representations is a list of matrices
param_name2label = {}

# first, collect probe representations across conditions, and steps
project_name = __name__
summaries = []
steps = set()
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         runs_path=RUNS_PATH,
                                         ludwig_data_path=LUDWIG_DATA_PATH,
                                         label_n=True):

    job_names = [path.stem for path in param_path.glob('*-*-*:*:*_*')]
    print(f'Found job names:')
    print(job_names)

    param_name2label[param_path.name] = label

    print(f'Collecting {WHERE} probe representations...')
    for row_id, job_name in enumerate(job_names):
        pattern = 'probe_reps_*.npz'
        npz_paths = list(sorted((param_path / job_name / 'saves').glob(pattern)))
        if not npz_paths:
            raise FileNotFoundError(f'Did not find {pattern} in {param_path}')

        # get representations
        for npz_path in npz_paths:
            with np.load(npz_path, allow_pickle=True) as loaded:
                if WHERE == 'input':
                    probe_reps = loaded['probe_reps_inp']
                else:
                    probe_reps = loaded['probe_reps_out']

            # collect
            step = int(npz_path.stem[-12:])
            step2param_name2probe_reps_list[step].setdefault(param_path.name, []).append(probe_reps)

print('Done collecting')

# to view how singular values changes in auto-updating plot
plt.ion()

# init a single figure
fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
ax.set_title(f'SVD of probe representations'
             f'\nat {WHERE}',
             y=1.25,
             fontdict={'fontsize': 6},
             )
ax.set_xlabel(f'Singular Dimension (1-{NUM_SINGULAR_DIMS})')
ax.set_ylabel(f'Proportion of Variance explained\n+/- Std')
ax.set_ylim(Y_LIMS)
x = np.arange(NUM_SINGULAR_DIMS)
ax.set_xticks(x)
ax.set_xticklabels([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)

# create camera for animation
camera = Camera(fig)

# do the computation and animation
for step, param_name2probe_reps_list in step2param_name2probe_reps_list.items():

    print(f'Computing results at step={step:>12,}')

    lines = []
    for param_name, probe_reps_list in param_name2probe_reps_list.items():

        # do svd for each job (get s_mat)
        s_mat = np.zeros((len(probe_reps_list), NUM_SINGULAR_DIMS))
        for row_id, probe_reps in enumerate(probe_reps_list):
            s = np.linalg.svd(probe_reps, compute_uv=False)
            s = (s / np.sum(s))[:NUM_SINGULAR_DIMS]
            s_mat[row_id] = s

        # plot one line
        y_mean = s_mat.mean(axis=0)
        y_std = s_mat.std(axis=0)
        x = np.arange(len(y_mean))
        label = param_name2label[param_name]
        color = 'C0' if 'reverse=False' in label else 'C1'
        line, = ax.plot(x, y_mean, color=color, label=label if step == 0 else '__nolegend')
        lines.append(line)
        ax.fill_between(x,
                        y_mean + y_std / 2,
                        y_mean - y_std / 2,
                        alpha=0.2,
                        color=color)
    fig.tight_layout()

    # update figure to show in interactive window
    # note: turn off Python Scientific in Pycharm Settings
    fig.canvas.draw()
    fig.canvas.flush_events()

    camera.snap()

    # removing lines after snap() results in animation error.
    # removing lines before snap makes interactive window work as expected, but removes lines from animation altogether
    # for line in lines:
    #     line.remove()

# create legend manually, once, so it is not plotted multiple times
# note: even though plotting legend at end results in not show up in interactive window, it shows up correctly in gif
ax.legend(bbox_to_anchor=(0.5, 1.0),
          borderaxespad=1.0,
          fontsize=configs.Figs.leg_fs,
          frameon=False,
          loc='lower center',
          ncol=3,
          )

animation = camera.animate()
fn = f'test.gif'
animation.save(str(configs.Dirs.animations / fn))