

# specify params to submit here
param2requests = {
    'reverse': [True, False],
    'legacy': [False],
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
    'num_types': 4096,
    'slide_size': 3,  # 3 is equivalent to approximately 20 iterations when batch_size=64
    'context_size': 7,  # number of backprop-through-time steps
    'batch_size': 64,
    'flavor': 'srn',  # simple-recurrent
    'hidden_size': 512,
    'lr': 0.01,
    'optimizer': 'adagrad',
}