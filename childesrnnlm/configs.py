from pathlib import Path


class Dirs:
    root = Path(__file__).parent.parent
    src = Path(__file__).parent
    structures = root / 'data' / 'structures'
    summaries = root / 'summaries'
    runs = root / 'runs'
    animations = root / 'animations'


class EntropicStart:
    num_left_words = 5
    num_right_words = 1


class Eval:
    train_pp = False  # extremely slow if True
    structures = ['sem-2021']
    num_steps_to_eval = 100_000
    high_res_eval_steps = [0, 1_000, 10_000, 20_000, 30_000, 40_000, 50_000, 60_000, 70_000, 80_000, 90_000,
                           100_000, 110_000, 120_000, 130_000, 150_000, 160_000, 170_000, 180_000, 190_000]

    frequency_weighting = True  # TODO try False
    min_num_test_tokens = 100_000
    ws_max_rows = 128
    as_max_rows = 128  # all 700 probes with 8K vocab takes 15min, but 128 probes require only 20 secs

    directions = ['l', 'c']  # 'r' is allowed but rarely useful,
    # because right-context representations reflect distributions that come after right-contexts,
    # and thus have little to do with probes
    locations = ['inp', 'out']
    context_types = ['n', 'o']

    # set to True to perform an evaluation
    calc_ba = bool(1)  # balanced accuracy
    calc_si = bool(0)  # silhouette score
    calc_sd = bool(0)  # S-dbw score
    calc_ma = bool(0)  # vector magnitude
    calc_pr1 = bool(0)  # divergence between actual and theoretical prototype
    calc_pr2 = bool(0)  # divergence between exemplars and theoretical prototype
    calc_pd = bool(0)  # pairwise divergences  - does not replicate findings from osf.io paper (2019)
    calc_pe = bool(0)  # pairwise euclidean distances
    calc_cs = bool(0)  # cosine similarity
    calc_cc = bool(0)  # cosine similarity within each category
    calc_op = bool(0)  # distance of prototype at input to origin
    calc_en = bool(0)  # entropy
    calc_eo = bool(0)  # entropy of representation of origin
    calc_fr = bool(1)  # fragmentation
    calc_cd = bool(0)  # within-probe divergence of contextualized representations
    calc_ds = bool(1)  # divergence from superordinate

    max_num_exemplars = 8192  # keep this as large as possible to reproduce age-order effect


class Figs:
    lw = 1
    axlabel_fs = 12
    leg_fs = 6
    dpi = 300
