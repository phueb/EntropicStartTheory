

# specify params to submit here
param2requests = {
    'reverse': [True, False],
    'context_size': [1],  # context-size = 1 is only way to decrease distance-to-prototype over time
}

param2debug = {
    'context_size': 1,
    'slide_size': 64,
    'batch_size': 64,
}

# default params
param2default = {
    'reverse': False,
    'shuffle_docs': False,   # this is an important control (contents of parts are randomly chosen)
    'corpus': 'childes-20191112',
    'ba_names': [],
    'dp_names': ['singular-nouns-4096', 'all-verbs-4096'],
    'num_types': 4096,
    'slide_size': 3,  # 3 is equivalent to approximately 20 iterations when batch_size=64
    'context_size': 7,  # default: 7 (equivalent to number of backprop-through-time steps)
    'batch_size': 64,
    'flavor': 'srn',  # srn, lstm
    'hidden_size': 512,  # default: 512
    'lr': 0.01,
    'optimizer': 'adagrad',
}