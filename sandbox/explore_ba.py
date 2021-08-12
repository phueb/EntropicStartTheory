import numpy as np
from typing import Optional
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.stats import sem, t
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from ludwig.results import gen_param_paths
from categoryeval.ba import BAScorer


from childesrnnlm import __name__, configs
from childesrnnlm.io import load_probe2cat
from childesrnnlm.params import param2default, param2requests
from childesrnnlm.figs import make_summary_fig

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH: Optional[Path] = None  # configs.Dirs.runs if loading runs locally or None if from ludwig

WHERE = ['input', 'output'][0]
FIRST_NUM_STEPS = 100  # the number of first training steps (otherwise, script takes long)
METRIC = 'euclidean'

# init collection of ba
param_name2ba_mat = {}  # matrix has shape (num jobs, num steps)

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

        # truncate steps to shorten waiting time
        num_steps = min(len(npz_paths), FIRST_NUM_STEPS)
        npz_paths = npz_paths[:num_steps]

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

            # compute balanced accuracy - convenient for offline exploration
            TARGET_STRUCTURE = 'sem-2021'
            probe2cat = load_probe2cat(configs.Dirs.root, TARGET_STRUCTURE, 'aochildes')
            ba_scorer = BAScorer(probe2cat)
            if METRIC == 'cosine':
                sim_mat = cosine_similarity(probe_reps)
            elif METRIC == 'euclidean':
                sim_mat = 1 / (euclidean_distances(probe_reps) + 1e-6)  # TODO is this right?
            else:
                raise AttributeError('Invalid arg to METRIC')

            # TODO ba scorer only searches sim threshold range of 0 to 1.o but euclidean sim range extends byond 1.0
            ba, th = ba_scorer.calc_score(sim_mat, ba_scorer.probe_store.gold_sims, 'ba',
                                      return_threshold=True)
            print(f'best threshold={th}')

            # collect ba
            param_name2ba_mat.setdefault(param_path.name, np.zeros((len(job_names), len(npz_paths))))
            param_name2ba_mat[param_path.name][row_id][col_id] = ba
            print(param_name2ba_mat[param_path.name].round(2))

    # collect summary for each param_name
    job_id = None
    x = np.array(list(sorted(list(steps))))
    y_mean = param_name2ba_mat[param_path.name].mean(axis=0)
    y_sem = sem(param_name2ba_mat[param_path.name], axis=0)
    confidence = 0.95
    n = len(job_names)
    h = y_sem * t.ppf((1 + confidence) / 2, n - 1)  # margin of error
    summary = (x, y_mean, h, label, job_id)
    summaries.append(summary)  # summary contains: x, mean_y, margin-of-error, label, job_id

if WHERE == 'output':
    y_lims = [0.5, 1]
else:
    y_lims = [0.5, 1]


y_label = f'Balanced Accuracy' \
          f'\nProbe Representations at {WHERE}' \
          f'\nsimilarity metric={METRIC}' \
          f'\n+/- 95%-CI'
fig = make_summary_fig(summaries,
                       ylabel=y_label,
                       figsize=(6, 4),
                       ylims=y_lims,
                       )
fig.show()