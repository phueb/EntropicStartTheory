from dataclasses import dataclass
from typing import Tuple, Union

# specify params to submit here
param2requests = {
    'reverse': [True, False],

    'flavor': ['srn'],

    'corpus': ['aonewsela'],
    'num_types': [6_000],

    # 'start': ['entropic', 'singleton', 'none'],

}

param2debug = {
    'corpus': 'aonewsela',
    'context_size': 2,
    'num_iterations': (1, 1),
    'num_transcripts': 100,
}

# default params
param2default = {
    'shuffle_transcripts': False,
    'corpus': 'aochildes',  # or aonewsela
    'num_types': 8000,  # lower than 8K preserves age-order effect but reduces balanced accuracy
    'num_transcripts': None,  # useful for debugging only
    'num_parts': 8,  # the lower the better performance, and age-order effect occurs across num_parts=2-256
    'context_size': 7,  # number of backprop-through-time steps, 7 is better than lower or higher
    'start': 'none',

    'flavor': 'srn',  # simple-recurrent
    'hidden_size': 512,
    'num_layers': 1,
    'bias': True,

    'sliding': False,
    'reverse': False,
    'num_iterations': (12, 12),  # more or less than 12 is worse
    'batch_size': 64,
    'lr': 0.01,
    'optimizer': 'adagrad',

}


@dataclass
class Params:
    """
    this object is loaded at the start of job.main() by calling Params.from_param2val(),
    and is populated by Ludwig with hyper-parameters corresponding to a single job.
    """
    shuffle_transcripts: bool
    corpus: str
    num_types: int
    num_transcripts: Union[None, int]
    num_parts: int
    context_size: int
    start: str

    flavor: str
    hidden_size: int
    num_layers: int
    bias: bool

    reverse: bool
    sliding: bool
    num_iterations: Tuple[int, int]
    batch_size: int
    lr: float
    optimizer: str

    @classmethod
    def from_param2val(cls, param2val):
        """
        instantiate class.
        exclude keys from param2val which are added by Ludwig.
        they are relevant to job submission only.
        """
        kwargs = {k: v for k, v in param2val.items()
                  if k not in ['job_name', 'param_name', 'save_path', 'project_path']}
        return cls(**kwargs)
