from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    src = Path(__file__).parent
    structures = root / 'data' / 'structures'
    summaries = root / 'summaries'
    runs = root / 'runs'


class Start:
    num_left_words = 5
    num_right_words = 1


class Eval:
    train_pp = False  # extremely slow if True
    structures = ['sem-2021']
    num_steps_to_eval = 100_000
    high_res_eval_steps = [0, 1_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000,
                           100_000, 110_000, 120_000, 130_000, 150_000, 160_000, 170_000, 180_000, 190_000]

    # TODO reduce min_num_test_tokens

    min_num_test_tokens = 200_000
    ws_max_rows = 128
    as_max_rows = 128  # all 700 probes with 8K vocab takes 15min, but 128 probes require only 20 secs

    # set to True to calculate an evaluation metric
    calc_ma = bool(1)  # vector magnitude
    calc_ra = bool(0)  # raggedness of in-out mapping
    calc_ba = bool(1)  # balanced accuracy
    calc_ws = bool(1)  # within-category spread
    calc_as = bool(1)  # across-category spread  - does not replicate findings from osf.io paper (2019)
    calc_di = bool(1)  # Euclidean and cosine distance
    calc_dp = bool(1)  # divergence from prototype
    calc_du = bool(1)  # divergence from unigram prototype
    calc_si = bool(1)  # silhouette score
    calc_sd = bool(1)  # S-dbw score
    calc_pi = bool(1)  # distance of prototype at input to origin at input
    calc_ep = bool(1)  # entropy of probe representations at output
    calc_fr = bool(1)  # fragmentation of probe representations

    max_num_exemplars = 8192  # keep this as large as possible to reproduce age-order effect


class Figs:
    lw = 1
    axlabel_fs = 12
    leg_fs = 6
    dpi = 300
