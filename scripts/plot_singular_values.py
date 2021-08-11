"""
Plot animation of singular values changing over training steps.

To see the animation, we cannot plot in the Pycharm tool window.
Turn this off in Setting -> Python Scientific
"""

import numpy as np
from typing import Optional
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import time

from ludwig.results import gen_param_paths

from childesrnnlm import __name__
from childesrnnlm.params import param2default, param2requests


LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH: Optional[Path] = None  # configs.Dirs.runs if loading runs locally or None if from ludwig

FIRST_NUM_STEPS = 100
WHERE = ['input', 'output'][0]
NUM_SINGULAR_DIMS = 64

if WHERE == 'output':
    Y_LIMS = [0, 0.5]
else:
    Y_LIMS = [0, 0.05]

# to view how singular values changes in auto-updating plot
plt.ion()


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

        num_steps = min(len(npz_paths), FIRST_NUM_STEPS)
        npz_paths = npz_paths[:FIRST_NUM_STEPS]

        # init plot once for each job
        fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
        ax.set_xlabel(f'Singular Dimension (1-{NUM_SINGULAR_DIMS})')
        ax.set_ylabel(f'Proportion of Variance explained')
        ax.set_ylim(Y_LIMS)
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top=False, right=False)

        colors = iter(cm.rainbow(np.linspace(0, 1, len(npz_paths))))

        # load representations at each step
        for col_id, npz_path in enumerate(npz_paths):

            print(f'Loading representations from archive at {npz_path.relative_to(param_path.parent)}')
            with np.load(npz_path, allow_pickle=True) as loaded:
                if WHERE == 'input':
                    probe_reps = loaded['probe_reps_inp']
                else:
                    probe_reps = loaded['probe_reps_out']

            step = int(npz_path.stem[-12:])
            steps.add(step)
            print(f'step={step}')

            if WHERE == 'input':
                time.sleep(1)  # to slow plotting

            # do svd
            s = np.linalg.svd(probe_reps, compute_uv=False)

            # TODO collect s across jobs and THEN plot average (not individual jobs)

            # plot
            # note: if plot does not show up, turn off Python Scientific in Pycharm Settings
            ax.set_title(f'SVD of probe representations'
                         f'\nat {WHERE}'
                         f'\nstep={step:,}'
                         )
            color = next(colors)
            y = (s / np.sum(s))[:NUM_SINGULAR_DIMS]
            line, = ax.plot(y, color=color)
            fig.canvas.draw()
            fig.canvas.flush_events()
