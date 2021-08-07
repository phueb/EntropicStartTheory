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
    high_res_eval_steps = [0, 1_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000]
    min_num_test_tokens = 0
    ws_max_rows = 1024
    as_max_rows = 1024
    cs_metric = 'js'  # is 3X faster than 'js' but not normalized
    ra_metric = 'js'

    # set to True to calculate an evaluation metric
    calc_ma = bool(1)  # vector magnitude
    calc_ra = bool(0)  # raggedness of in-out mapping # TODO not very useful - improve?
    calc_ba = bool(1)  # balanced accuracy
    calc_ws = bool(0)  # within-category spread
    calc_as = bool(1)  # across-category spread
    calc_dp = bool(1)  # divergence from prototype
    calc_du = bool(1)  # divergence from unigram prototype
    calc_si = bool(1)  # silhouette score
    calc_sd = bool(1)  # S-dbw score

    max_num_exemplars = 8192  # keep this as large as possible to reproduce age-order effect


class Figs:
    lw = 1
    axlabel_fs = 12
    leg_fs = 6
    dpi = 300
