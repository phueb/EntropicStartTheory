from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    src = Path(__file__).parent
    corpora = root / 'corpora'


class Global:
    train_pp = False
    legacy = True  # TODO remove this option in the future


class Eval:

    # Careful: make num_ts divisible by num_iterations

    num_ts = 20   # number of time points at which to evaluate performance
    stop_t = 20  # number of time points after which to exit (do not exit prematurely if None)
    dp_num_parts = 1
    num_test_docs = 0
    max_num_exemplars = 4096  # TODO this may need to be set higher for semantic categories


class Figs:
    lw = 2
    axlabel_fs = 14
    leg_fs = 12
    dpi = 163