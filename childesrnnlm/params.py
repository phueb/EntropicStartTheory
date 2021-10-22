"""
notes:

age-order effect for AO-Newsela with dedicated semantic structure (created in August 2021)
is most prominent with vocab size = 6K and 8K and disappears at 16K.
the effect also disappears with num_parts=4, and num_parts=16, and num_iterations=(32,4), and num_iterations=(20, 20)
"""
from dataclasses import dataclass
from typing import Tuple, Union, Dict, Any

# specify params to submit here
param2requests = {
    # 'reverse': [False, True],

    # 'reverse_tokens': [True, False],

    # paper 2 exp1a
    # 'corpus': [
        # 'axy',
        # 'yxb',
        # 'yxy-rule:and',
        # 'yxy-rule:or',
    # ],

    # paper 2 exp1b language PAY
    # 'corpus': [
    #     'rxy-redundant_with:y-redundancy:0.0',
    #     'rxy-redundant_with:y-redundancy:0.2',
    #     'rxy-redundant_with:y-redundancy:0.4',
    #     'rxy-redundant_with:y-redundancy:0.6',
    #     'rxy-redundant_with:y-redundancy:0.8',
    #     'rxy-redundant_with:y-redundancy:1.0',
    # ],

    # paper 2 exp1b language DAX
    'corpus': [
        'rxy-redundant_with:x-prop_fully_determined:0.0',
        'rxy-redundant_with:x-prop_fully_determined:0.2',
        'rxy-redundant_with:x-prop_fully_determined:0.4',
        'rxy-redundant_with:x-prop_fully_determined:0.6',
        'rxy-redundant_with:x-prop_fully_determined:0.8',
        'rxy-redundant_with:x-prop_fully_determined:1.0',
    ],

    # paper 2 exp1b language DAY
    # 'corpus': [
    #     'rxy-redundant_with:y-prop_fully_determined:0.0',
    #     'rxy-redundant_with:y-prop_fully_determined:0.2',
    #     'rxy-redundant_with:y-prop_fully_determined:0.4',
    #     'rxy-redundant_with:y-prop_fully_determined:0.6',
    #     'rxy-redundant_with:y-prop_fully_determined:0.8',
    #     'rxy-redundant_with:y-prop_fully_determined:1.0',
    # ],


    # paper 3 experiment 3a
    # 'num_parts': [1],
    # 'num_iterations': [
    #     (40, 40),
    # ],
    # 'probe_embeddings_info': [
    #     ('axy', 'sem-2021',  37_000),
    #     (None, None, None),
    #     ],

    # paper 3 experiment 3b
    # 'num_parts': [1],
    # 'num_iterations': [
    #     (30, 30),
    #     (40, 40),
    # ],
    # 'probe_embeddings_info': [
    #     ('param_001', 'sem-2021',  1_000_000),
    #     (None, None, None),
    # ],


}

param2debug = {
    # 'corpus': 'aonewsela',
    'context_size': 4,
    'num_iterations': (1, 1),
    'num_transcripts': 300,
}

# default params
param2default = {
    'shuffle_transcripts': False,
    'corpus': 'aochildes',  # or aonewsela, or
    'num_types': 8000,  # lower than 8K preserves age-order effect but reduces balanced accuracy
    'num_transcripts': None,  # useful for debugging only
    'num_parts': 8,  # the lower the better performance, and age-order effect occurs across num_parts=2-256
    'context_size': 7,  # number of backprop-through-time steps, 7 is better than lower or higher
    'start': 'none',  # unused (legacy) parameter
    'shuffle_at_start': True,

    'flavor': 'srn',  # simple-recurrent
    'hidden_size': 512,
    'num_layers': 1,
    'bias': True,
    'probe_embeddings_info': (None, None, None),  # pram_name, structure_name, step (of type int)

    'sliding': False,
    'reverse_tokens': False,
    'reverse': False,
    'num_iterations': (12, 12),  # more or less than 12 is worse when num_parts approx 8
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
    shuffle_at_start: bool

    flavor: str
    hidden_size: int
    num_layers: int
    bias: bool
    probe_embeddings_info: Dict[str, Any]

    reverse: bool
    reverse_tokens: bool
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
