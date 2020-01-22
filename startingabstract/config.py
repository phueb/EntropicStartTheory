from pathlib import Path


class LocalDirs:
    root = Path(__file__).parent.parent
    src = Path(__file__).parent
    runs = root / 'runs'


class Global:
    train_pp = False


class Symbols:
    OOV = 'OOV'


class Eval:
    num_evaluations = 100


class Metrics:
    ba_o = 'ba_ordered'
    ba_n = 'ba_none'
    an_nouns = 'an_nouns'  # abstractness of noun representation
    an_nouns_std = 'an_nouns_std'
    an_vocab = 'an_vocab'  # abstractness of all words - just unigram frequency
    an_vocab_std = 'an_vocab_std'


class Figs:
    lw = 2
    axlabel_fs = 12
    leg_fs = 10
    dpi = None