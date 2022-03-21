from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    src = Path(__file__).parent
    corpora = root / 'data' / 'corpora'
    structures = root / 'data' / 'structures'


class Start:
    num_left_words = 5
    num_right_words = 1


class Eval:
    train_pp = False  # extremely slow if True
    num_steps_to_eval = 50_000
    min_num_test_tokens = 0
    cs_max_rows = 128

    ba_o = True
    ba_n = True
    si_o = True
    si_n = True
    sd_o = True
    sd_n = True

    max_num_exemplars = 8192  # keep this as large as possible to reproduce age-order effect


class Figs:
    lw = 1
    axlabel_fs = 12
    leg_fs = 6
    dpi = 163