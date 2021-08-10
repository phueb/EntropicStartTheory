import numpy as np
from typing import Optional
from pathlib import Path

from ludwig.results import gen_param_paths

from childesrnnlm import __name__, configs
from childesrnnlm.params import param2default

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH: Optional[Path] = configs.Dirs.runs  # configs.Dirs.runs if loading runs locally or None if from ludwig

param2requests = {
    'corpus': ['aonewsela'],
    'context_size': [2],
    'num_iterations': [(1, 1)],
    'num_transcripts': [100],
}

project_name = __name__
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         runs_path=RUNS_PATH,
                                         ludwig_data_path=LUDWIG_DATA_PATH,
                                         label_n=False):

    # load representations
    fn = 'probe_reps_000000000000.npz'
    npz_paths = list(param_path.rglob(fn))
    if not npz_paths:
        raise FileNotFoundError(f'Did not find {fn} in {param_path}')
    npz_path = npz_paths[-1]
    print(f'Loading representations from archive at {npz_path}')
    with np.load(npz_path, allow_pickle=True) as loaded:
        probe_reps_inp = loaded['probe_reps_inp']
        probe_reps_out = loaded['probe_reps_out']

        print(probe_reps_inp.shape)
        print(probe_reps_out.shape)