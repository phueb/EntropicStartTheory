"""
How much information is there in the left-context or right-context of a target word about semantic category membership?

The LDA classifier achieves 100% accuracy with default prams and no scaling.
The Logistic classifier achieves above 90% accuracy only if input features are scaled.
Perfect performance is achieved because there are only 700 targets and there are many more features (5k),
so the classifier can remember the training data perfectly.
also, these accuracies are obtained when tokens are collapsed to types,
and this makes each data point much more informative than individual tokens


It takes about 600 of the most frequent contexts to get above 95% accuracy and about 1000 to get 99% accuracy.
This is only true when contexts must not be shared across the 2 partitions.
When context sharing is enforced, it takes more than 900 contexts just to get above 90%

The out-of-sample accuracy of classifying right contexts might best be understood as
the quality of the supervisory signal for the RNN.
ie. how diagnostic are the distributions that best define each category?
there are 28 distributions, one for each category. they are learned by LDA

The test split includes all of the same probes, but the random partition used to collect frequency was different.
When holding out probe types instead of a random partition, generalization accuracy is worse.

"""
import random
from dataclasses import dataclass
from typing import List
from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, auc
from pyitlib import discrete_random_variable as drv

from aochildes.dataset import AOChildesDataSet

from entropicstarttheory import configs
from entropicstarttheory.io import load_probe2cat
from entropicstarttheory.bpe import train_bpe_tokenizer

NUM_REPS = 10
SHUFFLE_SENTENCES = True
BPE_VOCAB_SIZE = 8_000  # this number includes 256 reserved latin and non-latin characters
N_COMPONENTS = 27  # should be number of categories - 1
NUM_X = 20  # the more, the smoother the lines in the figure
STRUCTURE_NAME = 'sem-all'
INCLUDE_SHUFFLED_LABELS_CONDITION = True

VERBOSE_COEF = False
VERBOSE_AUC = False
VERBOSE_MUTUAL_INFO = True

if BPE_VOCAB_SIZE < 256:
    raise AttributeError('BPE_VOCAB_SIZE must be greater than 256.')

if NUM_REPS > 1 and not SHUFFLE_SENTENCES:
    raise RuntimeError('Using more than one replication requires shuffling the sentences')

np.set_printoptions(suppress=True)

num_parts = 2
num_contexts_levels = [int(i) for i in np.linspace(30, 500, NUM_X)]  # approx 100-200


def save_to_text(fn: str,
                 index: List,
                 means: np.array,
                 stds: np.array,
                 ) -> None:  # for overleaf data
    # make path
    path = configs.Dirs.summaries / f'{fn}.txt'  # txt format makes file content visible on overleaf.org
    if not path.parent.exists():
        path.parent.mkdir()

    # save to text
    df = pd.DataFrame(data={'mean': means, 'std': stds}, index=index)
    df.index.name = 'num_contexts'
    df.round(4).to_csv(path, sep=' ')


def train_clf(x_: np.array,
              y_: np.array,
              ):

    clf_ = LinearDiscriminantAnalysis(n_components=N_COMPONENTS)
    clf_.fit(x_, y_)

    return clf_  # for testing on another partition


def make_x_y(df_x_train_: pd.DataFrame,
             contexts_shared_: List[str],
             shuffled_labels_: bool,
             rep_: int,
             ):

    # scale before feature selection so that values are the same no matter what features are selected.
    # note: this was observed not to make much of a difference in accuracy
    # df_x = (df_x.T / df_x.T.sum()).T

    # get only features that are shared
    df_x_train_ = df_x_train_[contexts_shared_]

    assert df_x_train_.columns.values.tolist() == contexts_shared_
    assert len(df_x_train_.columns.values.tolist()) == len(contexts_shared_)

    # convert data to numeric
    x_ = df_x_train_.values
    y_ = np.array([cat2id[probe2cat[p]] for p in df_x_train_.index])

    # shuffle labels
    # note: make sure that the same shuffled labels are used during training AND evaluation by setting random seed
    if shuffled_labels_:
        np.random.seed(rep_)
        y_ = np.random.permutation(y_)

    # scale (so that features are the proportion of times a probe occurs with a given context)
    x_ = (x_ + 1) / (x_.sum(1)[:, np.newaxis] + 1)

    return x_, y_


def eval_clf(clf_: LinearDiscriminantAnalysis,
             x_: np.array,
             y_: np.array,
             ) -> float:

    # eval on the same data
    acc_ = clf_.score(x_, y_)
    print(f'accuracy ={acc_:.4f}')

    # get tpr and fpr for each category
    prob_mat = clf_.predict_proba(x_)
    cat2roc = {}
    for cat in cat2id:
        fpr, tpr, thresholds = roc_curve(np.where(y_ == cat2id[cat], 1, 0),
                                         prob_mat[:, cat2id[cat]])
        cat2roc[cat] = (fpr, tpr)

    # print AUC for each category (labels and colors are sorted by AUC)
    for cat, (fpr, tpr) in sorted(cat2roc.items(),
                                  key=lambda i: auc(*i[1]),
                                  reverse=True):
        if VERBOSE_AUC:
            print(f'{cat:<12} auc={auc(fpr, tpr)}')

    return acc_


# ############################################################## conditions

context_directions = ['l', 'r']
shuffled_labels_levels = [True, False]
transfer_directions = ['f', 'b']  # forward, and backward transfer

# ############################################################## start tokenization

# get corpus
sentences = AOChildesDataSet().load_sentences()
if SHUFFLE_SENTENCES:
    random.shuffle(sentences)
text_original = ' '.join(sentences)
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
print(f'Tokenizing {len(sentences)} transcripts..', flush=True)
special_token = '<probe>'
tokenizer = train_bpe_tokenizer(sentences, BPE_VOCAB_SIZE, special_tokens=[special_token])
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

# ############################################################## get data


def get_data():

    # collect all co-occurrences
    name2col = {'probe': [],
                'l': [],
                'r': [],
                'cat': [],
                }
    for n, t in enumerate(tokens[:-1]):
        if t in probe2cat:
            name2col['probe'].append(t)
            name2col['l'].append(tokens[n - 1])
            name2col['r'].append(tokens[n + 1])
            name2col['cat'].append(probe2cat[t])
    df = pd.DataFrame(data=name2col)
    df.set_index('probe', inplace=True)
    print(f'Number of probe occurrences={len(df)}')

    # shuffle or preserve age-order
    if NUM_REPS > 1 and SHUFFLE_SENTENCES:
        df = df.sample(frac=1.0, replace=False)

    # partition into 2 halves
    half = len(df) // 2
    df1 = df[:half]
    df2 = df[half:]

    print(f'Partitioned co-occurrence data {"randomly" if NUM_REPS > 1 and SHUFFLE_SENTENCES else "by age"}')

    # approximate the redundancy between left and right contexts and semantic category membership.
    # note: an upper bound on redundancy is the minimum mutual information, min(mi(l,c), mi(r,c))
    # Nils Bertschinger, Johannes Rauh, Eckehard Olbrich, and Jürgen Jost (2013).
    # Shared information—new insights and problems in decomposing information in complex systems.
    for pi, df_part in enumerate([df1, df2]):

        if VERBOSE_MUTUAL_INFO:
            token2id = {t: n for n, t in enumerate(tokens)}
            l_obs = [token2id[t] for t in df_part['l']]
            r_obs = [token2id[t] for t in df_part['r']]
            c_obs = [cat2id[cat] for cat in df_part['cat']]
            nmi_lc = drv.information_mutual_normalised(l_obs, c_obs)
            nmi_rc = drv.information_mutual_normalised(r_obs, c_obs)
            print(f'part_id={pi} nmi(l,c)               ={nmi_lc:.4f}')
            print(f'part_id={pi} nmi(r,c)               ={nmi_rc:.4f}')

    # transform data so that columns are contexts, rows are probes, and values are frequencies
    part_id2cd2df_x_ = defaultdict(dict)
    for pi, df_part in enumerate([df1, df2]):
        for cd in context_directions:
            dummies = pd.get_dummies(df_part[cd])
            df_x = dummies.groupby(dummies.index).sum()
            part_id2cd2df_x_[pi][cd] = df_x

            print(f'Number of columns in df={len(df_x.columns):>9,} with direction={cd}')

    # get contexts shared across partitions
    cd2context2f_ = defaultdict(Counter)
    for cd in context_directions:
        df_x1 = part_id2cd2df_x_[0][cd]
        df_x2 = part_id2cd2df_x_[1][cd]
        contexts1 = set(df_x1.columns)
        contexts2 = set(df_x2.columns)
        contexts_shared = set()
        contexts_shared.update(contexts2)
        contexts_shared.intersection_update(contexts1)
        t2f1 = df_x1[contexts_shared].sum(0).to_dict()
        t2f2 = df_x2[contexts_shared].sum(0).to_dict()
        cd2context2f_[cd].update(t2f1)
        cd2context2f_[cd].update(t2f2)

    return part_id2cd2df_x_, cd2context2f_

# ############################################################## train classifier on all conditions


@dataclass
class Fit:
    clf: LinearDiscriminantAnalysis
    context_dir: str
    num_contexts: int
    part_id: int
    shuffled_labels: bool

    # collecting the already processed data speeds evaluation
    x_train: np.array
    y_train: np.array

    x_valid: np.array
    y_valid: np.array

    rep: int


fits = []

for rep in range(NUM_REPS):

    # each replication uses a new random partitioning (for the purpose of getting error bars)
    part_id2cd2df_x, cd2context2f = get_data()

    # collect data and train
    for context_dir in context_directions:

        for part_id in range(num_parts):

            df_x_train = part_id2cd2df_x[{0: 0, 1: 1}[part_id]][context_dir]
            df_x_valid = part_id2cd2df_x[{0: 1, 1: 0}[part_id]][context_dir]

            for num_contexts in num_contexts_levels:

                context_shared2f: Counter = cd2context2f[context_dir]
                contexts_train = [c for c, f in context_shared2f.most_common(num_contexts)]

                for shuffled_labels in shuffled_labels_levels:

                    if shuffled_labels and not INCLUDE_SHUFFLED_LABELS_CONDITION:
                        continue

                    # get train and valid data (for later)
                    x_train, y_train = make_x_y(df_x_train, contexts_train, shuffled_labels, rep)
                    x_valid, y_valid = make_x_y(df_x_valid, contexts_train, shuffled_labels, rep)

                    # train
                    clf = train_clf(x_train, y_train)

                    # collect classifier, data, and information about condition
                    fits.append(Fit(clf=clf,
                                    context_dir=context_dir,
                                    num_contexts=num_contexts,
                                    part_id=part_id,
                                    shuffled_labels=shuffled_labels,
                                    x_train=x_train,
                                    y_train=y_train,
                                    x_valid=x_valid,
                                    y_valid=y_valid,
                                    rep=rep,
                                    ))

# ############################################################## evaluate classifiers


def get_train_condition(cd: str,
                        pi: int,
                        sl: bool
                        ) -> str:
    """ there are 4 conditions, one for each curve that is plotted. the design is 2x2x2"""
    return '-'.join([cd, f'partition{pi+1}', 'shuffled' if sl else ''])


def get_valid_condition(cd: str,
                        td: str,
                        sl: bool
                        ) -> str:
    """ there are 4 conditions, one for each curve that is plotted. the design is 2x2x2"""
    return '-'.join([cd, td, 'shuffled' if sl else ''])


train_condition2acc_mat = {get_train_condition(cd, pi, sl): np.zeros((NUM_REPS, NUM_X))
                           for cd, pi, sl in product(context_directions,
                                                     range(num_parts),
                                                     shuffled_labels_levels)}

valid_condition2acc_mat = {get_valid_condition(cd, td, sl): np.zeros((NUM_REPS, NUM_X))
                           for cd, td, sl in product(context_directions,
                                                     transfer_directions,
                                                     shuffled_labels_levels)}

chance_acc = 0.07

# evaluate classifiers on in-sample data
for fit in fits:

    acc = eval_clf(fit.clf,
                   fit.x_train,
                   fit.y_train
                   )

    # collect accuracy
    row_id = fit.rep
    col_id = num_contexts_levels.index(fit.num_contexts)
    condition = get_train_condition(fit.context_dir, fit.part_id, fit.shuffled_labels)
    train_condition2acc_mat[condition][row_id, col_id] = acc

    if not INCLUDE_SHUFFLED_LABELS_CONDITION:
        condition = get_train_condition(fit.context_dir, fit.part_id, True)
        train_condition2acc_mat[condition][row_id, col_id] = chance_acc


# evaluate classifiers on out-of-sample data
for fit in fits:

    acc = eval_clf(fit.clf,
                   fit.x_valid,
                   fit.y_valid,
                   )

    # collect accuracy
    row_id = fit.rep
    col_id = num_contexts_levels.index(fit.num_contexts)
    transfer_dir = 'f' if fit.part_id == 0 else 'b'
    condition = get_valid_condition(fit.context_dir, transfer_dir, fit.shuffled_labels)
    valid_condition2acc_mat[condition][row_id, col_id] = acc

    if not INCLUDE_SHUFFLED_LABELS_CONDITION:
        condition = get_valid_condition(fit.context_dir, transfer_dir, True)
        valid_condition2acc_mat[condition][row_id, col_id] = chance_acc

# show accuracy in figure (and save results to txt)
for collapsed in [True, False]:  # whether to show part 1 and part 2 separately (2 lines vs. 1)

    # note: top-left has context_dir=left and in-sample accuracy
    # note: top-right has context_dir=right and in-sample accuracy
    # note: bottom-left has context_dir=left and out-of-sample accuracy
    # note: bottom-right has context_dir=right and out-of-sample accuracy
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 4), dpi=configs.Figs.dpi)
    fig.suptitle('Accuracy collapsed across partition and transfer-direction' if collapsed
                 else 'Accuracy by partition and transfer-direction',
                 y=1.0)
    x_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]

    # plot in-sample accuracy in top row of figure

    for ax, context_dir in zip(axes[0], context_directions):
        if context_dir == context_directions[0]:
            ax.set_ylabel('In-Sample\nClassification Accuracy',
                          fontsize=10)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top=False, right=False)
        ax.set_title('left context' if context_dir == 'l' else 'right context ')
        ax.set_yticks(x_ticks)
        ax.set_yticklabels(x_ticks)
        ax.yaxis.grid(True)
        ax.set_ylim([0, 1.0])
        # separately plot accuracy for each part_id
        if not collapsed:
            for part_id in range(num_parts):
                for shuffled_labels in shuffled_labels_levels:
                    condition = get_train_condition(context_dir, part_id, shuffled_labels)
                    curve_data = train_condition2acc_mat[condition]
                    curve = curve_data.mean(axis=0)
                    ax.plot(num_contexts_levels,
                            curve,
                            color=f'C{part_id}',
                            linestyle='--' if shuffled_labels else '-',
                            label=condition,
                            )
                    std_half = curve_data.std(axis=0) / 2
                    ax.fill_between(num_contexts_levels,
                                    curve + std_half,
                                    curve - std_half,
                                    alpha=0.2,
                                    )
        # combine accuracy across part_ids
        else:
            for shuffled_labels in shuffled_labels_levels:
                condition1 = get_train_condition(context_dir, 0, shuffled_labels)
                condition2 = get_train_condition(context_dir, 1, shuffled_labels)
                mat1 = train_condition2acc_mat[condition1]
                mat2 = train_condition2acc_mat[condition2]
                curve_data = np.vstack((mat1, mat2))
                curve = curve_data.mean(axis=0)
                ax.plot(num_contexts_levels,
                        curve,
                        label=f'shuffled-labels = {shuffled_labels}',
                        )
                std_half = curve_data.std(axis=0) / 2
                ax.fill_between(num_contexts_levels,
                                curve + std_half,
                                curve - std_half,
                                alpha=0.2,
                                )
                save_to_text(fn=f'lda-acc_shuffled-labels={int(shuffled_labels)}_context-dir={context_dir}_train',
                             index=num_contexts_levels,
                             means=curve_data.mean(axis=0),
                             stds=curve_data.std(axis=0),
                             )

    # plot out-of-sample accuracy in bottom row of figure
    for ax, context_dir in zip(axes[1], context_directions):
        ax.set_xlabel('Number of Context Words',
                      fontsize=10)
        if context_dir == context_directions[0]:
            ax.set_ylabel('Out-of-Sample\nClassification Accuracy',
                          fontsize=10)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(axis='both', which='both', top=False, right=False)
        ax.set_yticks(x_ticks)
        ax.set_yticklabels(x_ticks)
        ax.yaxis.grid(True)
        ax.set_ylim([0, 0.5])
        # separately plot accuracy for each transfer direction
        if not collapsed:
            for transfer_dir in transfer_directions:
                for shuffled_labels in shuffled_labels_levels:
                    condition = get_valid_condition(context_dir, transfer_dir, shuffled_labels)
                    curve_data = valid_condition2acc_mat[condition]
                    curve = curve_data.mean(axis=0)
                    ax.plot(num_contexts_levels,
                            curve,
                            color=f'C{transfer_directions.index(transfer_dir)}',
                            linestyle='--' if shuffled_labels else '-',
                            label=condition,
                            )
                    std_half = curve_data.std(axis=0) / 2
                    ax.fill_between(num_contexts_levels,
                                    curve + std_half,
                                    curve - std_half,
                                    alpha=0.2,
                                    )
        # combine transfer directions into a single curve
        else:
            for shuffled_labels in shuffled_labels_levels:
                condition1 = get_valid_condition(context_dir, 'f', shuffled_labels)
                condition2 = get_valid_condition(context_dir, 'b', shuffled_labels)
                mat1 = valid_condition2acc_mat[condition1]
                mat2 = valid_condition2acc_mat[condition2]
                curve_data = np.vstack((mat1, mat2))
                curve = curve_data.mean(axis=0)
                ax.plot(num_contexts_levels,
                        curve,
                        label=f'shuffled-labels = {shuffled_labels}',
                        )
                std_half = curve_data.std(axis=0) / 2
                ax.fill_between(num_contexts_levels,
                                curve + std_half,
                                curve - std_half,
                                alpha=0.2,
                                )
                save_to_text(fn=f'lda-acc_shuffled-labels={int(shuffled_labels)}_context-dir={context_dir}_valid',
                             index=num_contexts_levels,
                             means=curve_data.mean(axis=0),
                             stds=curve_data.std(axis=0),
                             )

    plt.show()

