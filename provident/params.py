

# specify params to submit here
param2requests = {
    'reverse': [True, False],  # TODO add shuffle
    # 'corpus': ['newsela'],  # TODO try newer childes corpus 20191206
    # 'num_types': [4096 * 4],
}

param2debug = {
    'context_size': 1,
    'num_iterations': (1, 1),
}

# default params
param2default = {
    'shuffle_sentences': False,
    'corpus': 'childes-20191112',
    'num_types': 4096,
    'num_parts': 256,
    'context_size': 7,  # number of backprop-through-time steps

    'flavor': 'srn',  # simple-recurrent
    'hidden_size': 512,

    'sliding': False,
    'reverse': False,
    'num_iterations': (16, 16),
    'batch_size': 64,
    'lr': 0.01,
    'optimizer': 'adagrad',

    'exclude_number_words': True,
}