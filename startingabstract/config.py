from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    src = Path(__file__).parent
    corpora = root / 'corpora'


class Global:
    train_pp = False


class Symbols:
    OOV = 'OOV'


class Eval:
    num_ts = 500   # number of time points at which to evaluate performance
    stop_t = 3  # number of time points after which to exit (do not exit prematurely if None)
    dp_num_parts = 2
    num_test_docs = 0


class Figs:
    lw = 2
    axlabel_fs = 14
    leg_fs = 12
    dpi = 163