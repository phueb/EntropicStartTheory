

# specify params to submit here
param2requests = {
    'reverse': [True, False],
    'slide_size': [3],
    'context_size': [1],  # context-size = 1 is only way to decrease "abstractness" over time
}

param2debug = {
    'context_size': 7,
    'slide_size': 64,
    'batch_size': 64,
}

# default params
param2default = {
    'reverse': False,
    'shuffle_docs': False,   # this is an important control (contents of parts are randomly chosen)
    'corpus': 'childes-20180319',
    'ba_names': ['sem-4096'],
    'dp_names': ['childes-20191112-nouns'],
    'num_types': 4096,
    'slide_size': 3,  # TODO test
    'context_size': 7,  # default: 7 (equivalent to number of backprop-through-time steps)
    'batch_size': 64,
    'flavor': 'srn',  # srn, lstm
    'hidden_size': 512,  # default: 512
    'lr': 0.01,
    'optimizer': 'adagrad',
}