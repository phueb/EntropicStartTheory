"""
Research question:
is the mutual information between last-context-word and next-word lower in partition 1?
if so, this means that the mapping that the model learns can handle variation more,
meaning it can handle divergent information and turn it into something useful by keeping it close to a prototype.
this is useful for answering whether age-ordered training
 helps "protect" representations from prematurely differentiating.
 this "protection" would result from learning a mapping from high-variance to low-variance

last-context-word NOUN next-word

"""

from pyitlib import discrete_random_variable as drv
from numpy.lib.stride_tricks import as_strided
import numpy as np

from categoryeval.ba import BAScorer
from preppy.legacy import TrainPrep

from startingabstract.docs import load_docs
from startingabstract import config


CORPUS_NAME = 'childes-20191112'
NOUNS_NAME = 'sem-4096'

corpus_path = config.Dirs.corpora / f'{CORPUS_NAME}.txt'
train_docs, _ = load_docs(corpus_path)

prep = TrainPrep(train_docs,
                 reverse=False,
                 num_types=4096,
                 num_parts=2,
                 num_iterations=[20, 20],
                 batch_size=64,
                 context_size=7,
                 num_evaluations=10,
                 )

ba_scorer = BAScorer(CORPUS_NAME,
                     probes_names=[NOUNS_NAME],
                     w2id=prep.store.w2id)

nouns = ba_scorer.name2store[NOUNS_NAME].types
print(f'num nouns={len(nouns)}')

for token_ids in [prep.store.token_ids[:prep.midpoint],
                  prep.store.token_ids[-prep.midpoint:]]:

    # windows
    token_ids_array = np.array(token_ids, dtype=np.int64)
    num_possible_windows = len(token_ids_array) - prep.num_tokens_in_window
    shape = (num_possible_windows, prep.num_tokens_in_window)
    windows = as_strided(token_ids_array, shape, strides=(8, 8), writeable=False)
    print(f'Matrix containing all windows has shape={windows.shape}')

    # noun windows
    row_ids = np.isin(windows[:, -2], ba_scorer.name2store[NOUNS_NAME].vocab_ids)
    noun_windows = windows[row_ids]
    print(f'num noun-windows={len(noun_windows)}')

    x = noun_windows[:, -3]  # last-context-word
    y = noun_windows[:, -1]  # next-word

    print(f'mi={drv.information_mutual(x, y)}')
    abc = np.arange(prep.store.num_types)
    print(f'xy={drv.entropy_conditional(x, y, Alphabet_X=abc, Alphabet_Y=abc)}')

    # conditional entropy (x, y) = how much information I have about X given Y
    # this should be less in partition 1 if mapping between y->x is really lossier

    # it makes sense that this is true, because partition 2 has more function words,
    # and they allow more precise predictions to be made.
    # also, given a period, it i shard to tell what came before, and because periods are more frequent
    # in partition 1, this makes the mapping from y to x lossier

    print()
