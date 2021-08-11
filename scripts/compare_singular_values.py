"""
Compare singular value plot at specific time in training between 2 or more conditions
"""
import numpy as np
from typing import Optional
from pathlib import Path
from matplotlib import pyplot as plt

from ludwig.results import gen_param_paths

from childesrnnlm import __name__, configs
from childesrnnlm.params import param2default, param2requests


LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH: Optional[Path] = None  # configs.Dirs.runs if loading runs locally or None if from ludwig

STEP = 200_000
WHERE = ['input', 'output'][0]
NUM_SINGULAR_DIMS = 16

if WHERE == 'output':
    Y_LIMS = [0, 0.5]
else:
    Y_LIMS = [0, 0.025]

# init collection
param_name2s_mat = {}
labels = []

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

    for row_id, job_name in enumerate(job_names):

        pattern = 'probe_reps_*.npz'
        npz_paths = list(sorted((param_path / job_name / 'saves').glob(pattern)))
        if not npz_paths:
            raise FileNotFoundError(f'Did not find {pattern} in {param_path}')

        try:
            npz_path = [path for path in npz_paths if int(path.stem[-12:]) == STEP][0]
        except IndexError:
            raise FileNotFoundError(f'Did not find representations at step {STEP}')

        print(f'Loading representations from archive at {npz_path.relative_to(param_path.parent)}')
        with np.load(npz_path, allow_pickle=True) as loaded:
            if WHERE == 'input':
                probe_reps = loaded['probe_reps_inp']
            else:
                probe_reps = loaded['probe_reps_out']

        # do svd
        s = np.linalg.svd(probe_reps, compute_uv=False)
        s = (s / np.sum(s))[:NUM_SINGULAR_DIMS]

        # collect
        param_name2s_mat.setdefault(param_path.name, np.zeros((len(job_names), len(s))))
        param_name2s_mat[param_path.name][row_id] = s

    labels.append(label)

# figure
fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
ax.set_title(f'SVD of probe representations'
             f'\nat {WHERE}'
             f'\nstep={STEP:,}',
             y=1.25,
             fontdict={'fontsize': 6},
             )
ax.set_xlabel(f'Singular Dimension (1-{NUM_SINGULAR_DIMS})')
ax.set_ylabel(f'Proportion of Variance explained\n+/- Std')
ax.set_ylim(Y_LIMS)
ax.set_xticks([])
ax.set_xticklabels([])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(axis='both', which='both', top=False, right=False)
# plot
for label, (param_name, s_mat) in zip(labels, param_name2s_mat.items()):
    y_mean = s_mat.mean(axis=0)
    y_std = s_mat.std(axis=0)
    x = np.arange(len(y_mean))
    color = 'C0' if 'reverse=False' in label else 'C1'
    ax.plot(x, y_mean, color=color, label=label)
    ax.fill_between(x,
                    y_mean + y_std / 2,
                    y_mean - y_std / 2,
                    alpha=0.2,
                    color=color)
# legend
plt.legend(bbox_to_anchor=(0.5, 1.0),
           borderaxespad=1.0,
           fontsize=configs.Figs.leg_fs,
           frameon=False,
           loc='lower center',
           ncol=3,
           )
fig.tight_layout()
plt.show()
