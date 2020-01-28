

# specify params to submit here
param2requests = {
    'reverse': [True, False],
    'legacy': [True],
    'shuffle_sentences': [False],
    'ba_probes': [('sem-4096', )],
    'dp_probes': [('singular-nouns-4096', 'unconditional')],
}

param2debug = {
    'context_size': 1,
    'slide_size': 64,
    'batch_size': 64,
}

# default params
param2default = {
    'legacy': False,
    'reverse': False,
    'shuffle_sentences': False,
    'corpus': 'childes-20191112',
    'ba_probes': ('syn-4096', 'sem-4096'),
    'dp_probes': ('singular-nouns-4096', 'all-verbs-4096', 'unconditional'),
    'num_types': 4096,
    'slide_size': 3,  # 3 is equivalent to approximately 20 iterations when batch_size=64
    'context_size': 7,  # number of backprop-through-time steps
    'batch_size': 64,
    'flavor': 'srn',  # simple-recurrent
    'hidden_size': 512,
    'lr': 0.01,
    'optimizer': 'adagrad',
}