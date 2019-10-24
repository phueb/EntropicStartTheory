

# specify params to submit here
param2requests = {
    'reverse': [False, True],
    'shuffle_docs': [False, True],
}

param2debug = {
    'context_size': 3,
    'slide_size': 2,
    'batch_size': 6,
}

# default params
param2default = {
    'reverse': False,
    'shuffle_docs': False,   # this is an important control (contents of parts are randomly chosen)
    'corpus': 'childes-20180319',
    'probes': 'sem-4096',
    'num_types': 4096,
    'slide_size': 3,  # TODO test
    'context_size': 7,  # default: 7 (equivalent to number of backprop-through-time steps)
    'batch_size': 64,
    'flavor': 'srn',  # srn, lstm
    'hidden_size': 512,  # default: 512
    'lr': 0.01,
    'optimizer': 'adagrad',
}