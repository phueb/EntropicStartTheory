"""
How much information is there in the left-context or right-context of a target word about semantic category membership?

The LDA classifier achieves 100% accuracy with default prams and no scaling.
The Logistic classifier achieves above 90% accuracy only if input features are scaled.
Perfect performance is achieved because there are only 700 targets and there are many more features (5k),
so the classifier can remember the training data perfectly.
also, these accuracies are obtained when tokens are collapsed to types,
and this makes each data point much more informative than individual tokens


It takes about 600 of the most frequent contexts to get above 95% accuracy and about 1000 to get 99% accuracy

"""
from typing import List, Optional
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc

from aochildes.dataset import AOChildesDataSet

from entropicstarttheory import configs
from entropicstarttheory.io import load_probe2cat
from entropicstarttheory.bpe import train_bpe_tokenizer

RANDOM_LABELS = False
NUM_PARTITIONS = 2
BPE_VOCAB_SIZE = 8_000  # this number includes 256 reserved latin and non-latin characters
N_COMPONENTS = 27  # should be number of categories - 1
MIN_CONTEXT_FREQ = 10
STRUCTURE_NAME = 'sem-all'

VERBOSE = False

if BPE_VOCAB_SIZE < 256:
    raise AttributeError('BPE_VOCAB_SIZE must be greater than 256.')

if MIN_CONTEXT_FREQ > BPE_VOCAB_SIZE:
    raise AttributeError('NUM_MOST_FREQUENT_CONTEXTS must be greater than BPE_VOCAB_SIZE.')

np.set_printoptions(suppress=True)


def train_and_eval(df_: pd.DataFrame,
                   direction_: str,
                   contexts_shared_: List[str],
                   clf_: Optional = None,  # trained classifier
                   ):

    # transform data so that columns are contexts, rows are probes, and values are frequencies
    dummies = pd.get_dummies(df_[direction_])
    dummies['probe'] = df_['probe']
    dummies.set_index('probe', inplace=True)
    df_x = dummies.groupby('probe').sum()

    # get only features that are shared
    df_x = df_x[contexts_shared_]
    # print(df_x)

    # convert data to numeric
    X_ = df_x.values.astype(np.int64)
    y_ = np.array([cat2id[probe2cat[p]] for p in df_x.index])

    # train classifier
    if clf_ is None:
        print('Fitting classifier...')
        clf_ = LinearDiscriminantAnalysis(n_components=N_COMPONENTS)
        if RANDOM_LABELS:
            clf_.fit(X_, np.random.permutation(y_))
        else:
            clf_.fit(X_, y_)

    # eval on the same data
    acc = clf_.score(X_, y_)
    print(f'direction={direction_:<12} accuracy ={acc:.4f}')

    # get tpr and fpr for each category
    prob_mat = clf_.predict_proba(X_)
    cat2roc = {}
    for cat in cat2id:
        fpr, tpr, thresholds = roc_curve(np.where(y_ == cat2id[cat], 1, 0),
                                         prob_mat[:, cat2id[cat]])
        cat2roc[cat] = (fpr, tpr)

    # print AUC for each category (labels and colors are sorted by AUC)
    for cat, (fpr, tpr) in sorted(cat2roc.items(),
                                  key=lambda i: auc(*i[1]),
                                  reverse=True):
        if VERBOSE:
            print(f'{cat:<12} auc={auc(fpr, tpr)}')

    return clf_  # for testing on another partition


# ############################################################## start tokenization

# get corpus
transcripts = AOChildesDataSet().load_transcripts()
text_original = ' '.join(transcripts)
tokens_original = text_original.split()

# load only probes that are in the data
probes_in_data = set()
num_probes = 0
types_in_sentences = set(tokens_original)
probe2cat = load_probe2cat(configs.Dirs.root, STRUCTURE_NAME, 'aochildes')
num_probes += len(probe2cat)
for probe in probe2cat.keys():
    if probe in types_in_sentences:
        probes_in_data.add(probe)
    else:
        print(f'probe={probe:<24} not in original data. Excluded.')
probes = list(probes_in_data)

# make cat2id
categories = set(sorted(probe2cat.values()))
cat2id = {cat: n for n, cat in enumerate(categories)}

# replace probes in corpus with special_token
special_token = '<probe>'
tokens_special = []
replaced_probes = []
for token in tokens_original:
    if token in probes_in_data:
        tokens_special.append(special_token)
        replaced_probes.append(token)
    else:
        tokens_special.append(token)

# tokenize corpus
print(f'Tokenizing {len(transcripts)} transcripts..', flush=True)
special_token = '<probe>'
tokenizer = train_bpe_tokenizer(transcripts, BPE_VOCAB_SIZE, special_tokens=[special_token])
text_special = ' '.join(tokens_special)
tokens: List[str] = [t for t in tokenizer.encode(text_special, add_special_tokens=True).tokens
                     if t not in {'Ġ', '', ' '}]
print(f'{len(set(tokens)):,} types in tokenized text', flush=True)
print(f'Tokenized text has {len(tokens) - len(tokens_original):,} more tokens than before tokenization')

# TODO tokenization produces tokens that resemble probes, but should not be considered probes:
# TODO ('Ġadv', 'ice',), ('Ġpleas', 'ant')
# TODO how to differentiate this "incidental" probes from real probes?

# print(sorted(tokenizer.get_vocab().items(), key=lambda i: i[1]))

# replace special token with original probes
replaced_probes_it = iter(replaced_probes)
tokens = [token if token != special_token else next(replaced_probes_it) for token in tokens]
assert len(list(replaced_probes_it)) == 0

# ############################################################## end tokenization

# make partitions
part_id2tokens = {}
for part_id in range(NUM_PARTITIONS):
    partition_size = len(tokens) // NUM_PARTITIONS
    start = part_id * partition_size
    part_id2tokens[part_id] = tokens[start:start + partition_size]

# get all contexts
part_id2contexts_l = defaultdict(list)
part_id2contexts_r = defaultdict(list)
for part_id in range(NUM_PARTITIONS):
    part = part_id2tokens[part_id]
    for n, token in enumerate(part):
        if token in probe2cat:
            part_id2contexts_l[part_id].append(part[n - 1])
            part_id2contexts_r[part_id].append(part[n + 1])

# get shared contexts
dir2contexts_shared = {'l': set(part_id2contexts_l[0]),
                       'r': set(part_id2contexts_r[0])}
for part_id in range(NUM_PARTITIONS):
    dir2contexts_shared['l'].intersection_update(part_id2contexts_l[part_id])
    dir2contexts_shared['r'].intersection_update(part_id2contexts_r[part_id])

# only use most frequent shared contexts
counter_l = Counter()
counter_r = Counter()
for part_id in range(NUM_PARTITIONS):
    counter_l.update(part_id2contexts_l[part_id])
    counter_r.update(part_id2contexts_r[part_id])
dir2contexts_shared['l'] = [c for c in dir2contexts_shared['l'] if counter_l[c] > MIN_CONTEXT_FREQ]
dir2contexts_shared['r'] = [c for c in dir2contexts_shared['r'] if counter_r[c] > MIN_CONTEXT_FREQ]

for direction, contexts_shared in dir2contexts_shared.items():
    print(f'direction={direction} num contexts={len(contexts_shared)}')

# collect data, train, and evaluate on train data
part_id2classifiers = {part_id: [] for part_id in range(NUM_PARTITIONS)}
part_id2df = {}
directions = ['l', 'r']
for part_id in range(NUM_PARTITIONS):

    # collect data
    name2col = {'probe': [],
                'l': [],
                'r': [],
                'cat': [],
                }
    part = part_id2tokens[part_id]
    for n, token in enumerate(part):
        if token in probe2cat:
            name2col['probe'].append(token)
            name2col['l'].append(part[n - 1])
            name2col['r'].append(part[n + 1])
            name2col['cat'].append(probe2cat[token])
    df = pd.DataFrame(data=name2col)

    # train and evaluate
    for direction in directions:
        clf = train_and_eval(df,
                             direction,
                             contexts_shared_=list(sorted(dir2contexts_shared[direction])),
                             )
        part_id2classifiers[part_id].append(clf)

    print()

    part_id2df[part_id] = df


# evaluate classifier trained on partition 1 but with data from partition 2
for direction in directions:
    train_and_eval(part_id2df[1],
                   direction,
                   contexts_shared_=list(sorted(dir2contexts_shared[direction])),
                   clf_=part_id2classifiers[0][directions.index(direction)],
                   )

# evaluate classifier trained on partition 2 but with data from partition 1
for direction in directions:
    train_and_eval(part_id2df[0],
                   direction,
                   contexts_shared_=list(sorted(dir2contexts_shared[direction])),
                   clf_=part_id2classifiers[1][directions.index(direction)],
                   )
