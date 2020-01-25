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
    num_ts = 200   # number of time points at which to evaluate performance
    stop_t = 20  # number of time points after which to exit (do not exit prematurely if None)
    dp_num_parts = 2
    num_test_docs = 100


class Metrics:
    # ba = balanced-accuracy
    ba_o = 'ba_ordered'
    ba_n = 'ba_none'


class Figs:
    lw = 2
    axlabel_fs = 12
    leg_fs = 10
    dpi = None