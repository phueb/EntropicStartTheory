from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    src = Path(__file__).parent
    corpora = root / 'data' / 'corpora'
    structures = root / 'data' / 'structures'


class Eval:
    train_pp = False
    structures = ['sem-2021']
    num_steps_to_eval = 50_000
    min_num_test_tokens = 0
    cs_max_rows = 128

    ba_o = True
    ba_n = True
    si_o = False
    si_n = False
    sd_o = False
    sd_n = False

    max_num_exemplars = 8192  # keep this as large as possible to reproduce age-order effect


class Figs:
    lw = 1
    axlabel_fs = 12
    leg_fs = 6
    dpi = 163