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
    num_evaluations = 200
    dp_num_parts = 4


class Metrics:
    # ba = balanced-accuracy
    ba_o = 'ba_ordered'
    ba_n = 'ba_none'


class Figs:
    lw = 2
    axlabel_fs = 12
    leg_fs = 10
    dpi = None