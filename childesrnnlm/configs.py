from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    src = Path(__file__).parent
    structures = root / 'data' / 'structures'


class Start:
    num_left_words = 5
    num_right_words = 1


class Eval:
    train_pp = False  # extremely slow if True
    structures = ['sem-2021']
    num_steps_to_eval = 100_000
    min_num_test_tokens = 0
    cs_max_rows = 32
    cs_metric = 'xe'  # is 3X faster than 'js' but not normalized
    ra_metric = 'xe'

    # set to True to calculate an evaluation metric
    calc_ra = True
    calc_ba = True
    calc_ws = True  # within-category spread
    calc_as = True  # across-category spread
    calc_dp = True
    calc_si = True
    calc_sd = True

    max_num_exemplars = 8192  # keep this as large as possible to reproduce age-order effect


class Figs:
    lw = 1
    axlabel_fs = 12
    leg_fs = 6
    dpi = 163