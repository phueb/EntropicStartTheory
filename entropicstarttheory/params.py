"""
Explanation of hyper-parameters.

CORPUS:

reverse: `True` reverses the order of the corpus.
shuffle_transcripts: `True` shuffles the order of AO-CHILDES transcripts (or any other documents part of a larger corpus).
corpus: `aochildes` will load an age-ordered American-English corpus from the CHILDES database, using the package `aochildes`.
num_types: `8000` is the number of unique BPE tokens in the model's vocabulary, including any probe words. 
num_transcripts: `None` will load all available transcripts from the corpus. use a lower number for debugging.
num_parts: the number of partitions of the corpus, used for corpus re-ordering. performance peaks if set to `1`. the age-order effect occurs for any value between 2 and 256.
context_size: the number of backprop-through-time steps. set to `7` for optimal performance (i.e balanced accuracy).
start: unused (legacy) parameter. set to `None`.
shuffle_at_start: determines whether sentences in the first partition should be shuffled. set to `True` to reduce any unwanted effect due to starting training of each model on the same exact data.

MODEL:

flavor: `srn` or `lstm`
hidden_size: `512`
num_layers: `1`
bias: `True`
probe_embeddings_info: determines which probes to use for evaluation of balanced accuracy. a 3-element with `param_name`, `structure_name`, and `step`

TRAINING:

sliding: `False` to cycle over corpus partitions during training. otherwise model iterates over the corpus in order, once.
reverse_tokens: `False`. otherwise, the order of tokens in each sentence is reversed, while preserving the order of corpus partitions.
reverse: `True` reverses the order of corpus partitions. 
num_iterations: `(12, 12)` to iterate over each corpus partition 12 times. 12 is optimal given 8 partitions.
batch_size: `64` to get optimal performance and speed given the default hyper perameters. 
lr: `0.01`
optimizer: `adagrad`
"""

from dataclasses import dataclass
from typing import Tuple, Union, Dict, Any

# specify params to submit here
param2requests = {
    
    # an example of how to specify params. 
    # note: all possible combinations of the parameter values below will be submitted to Ludwig.
    'reverse': [False, True],
    'shuffle_transcripts': [True, False],
    'num_parts': [1, 8],
    'reverse_tokens': [True, False],
    'flavor': ['srn'],

    # paper 2 exp1a
    # 'corpus': [
        # 'axy',
        # 'yxb',
        # 'yxy-rule:and',
        # 'yxy-rule:or',
    # ],

    # paper 2 exp1b language PAX
    # 'corpus': [
    #     'rxy-redundant_with:x-redundancy:0.0',
    #     'rxy-redundant_with:x-redundancy:0.2',
    #     'rxy-redundant_with:x-redundancy:0.4',
    #     'rxy-redundant_with:x-redundancy:0.6',
    #     'rxy-redundant_with:x-redundancy:0.8',
    #     'rxy-redundant_with:x-redundancy:1.0',
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
    # 'corpus': [
    #     'rxy-redundant_with:x-prop_fully_determined:0.0',
    #     'rxy-redundant_with:x-prop_fully_determined:0.2',
    #     'rxy-redundant_with:x-prop_fully_determined:0.4',
    #     'rxy-redundant_with:x-prop_fully_determined:0.6',
    #     'rxy-redundant_with:x-prop_fully_determined:0.8',
    #     'rxy-redundant_with:x-prop_fully_determined:1.0',
    # ],

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
    'corpus': 'aochildes',  # or aonewsela
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
    'probe_embeddings_info': (None, None, None),  # param_name, structure_name, step (of type int)

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
