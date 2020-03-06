from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    src = Path(__file__).parent
    corpora = root / 'corpora'


class Eval:
    train_pp = False

    ba_probes = ('sem-4096',)
    cs_probes = ()
    dp_probes = ()

    num_total_ticks = 128   # number of time points at which to evaluate performance
    num_start_ticks = 16
    tick_step = 8  # skip this number of ticks until performance evaluation when not at start of training

    cs_max_rows = 128

    ba_o = False
    ba_n = True

    num_test_docs = 100
    max_num_exemplars = 8192  # keep this as large as possible to reproduce age-order effect


class Figs:
    lw = 2
    axlabel_fs = 14
    leg_fs = 12
    dpi = 163