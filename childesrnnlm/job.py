import time
import pyprind
import pandas as pd
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

from preppy import Prep

from childesrnnlm import configs
from childesrnnlm.io import load_probe2cat
from childesrnnlm.evaluation import update_ba_performance
from childesrnnlm.evaluation import update_pp_performance

from childesrnnlm.params import Params
from childesrnnlm.rnn import RNN


def main(param2val):
    # params
    params = Params.from_param2val(param2val)
    print(params)

    project_path = Path(param2val['project_path'])

    # load corpus
    corpus_path = project_path / 'data' / 'corpora' / f'corpus_{params.corpus}_{params.num_types}.txt'
    if not corpus_path.exists():
        raise FileNotFoundError(f'Did not find {corpus_path}')
    with corpus_path.open('r') as f:
        text_original = f.read().replace('\n', ' ')
    tokens_original = text_original.split()
    print(f'Loaded {len(tokens_original):,} words.')

    print(tokens_original[:100])

    # tokenize text
    tokens = tokens_original
    print(f'{len(set(tokens)):,} types in tokenized text', flush=True)

    # prepare data for batching
    prep = Prep(tokens,
                reverse=params.reverse,
                sliding=params.sliding,
                num_parts=params.num_parts,
                num_iterations=params.num_iterations,
                batch_size=params.batch_size,
                context_size=params.context_size,
                shuffle_within_part=False,
                min_num_test_tokens=configs.Eval.min_num_test_tokens,
                disallow_non_ascii=False,
                )

    # prepare artificially generated start sequences for batching
    if params.start != 'none':
        raise NotImplementedError
    else:
        prep_start = None
        print(f'Not adding start.')

    # combine start sequences and regular sequences
    if prep_start:
        raise NotImplementedError
    else:
        batch_generator = prep.generate_batches()
        high_resolution_eval_steps = [0]
        num_train_mbs = prep.num_mbs

    # load all structures, for evaluation, each consisting of a dict mapping probe -> category,
    # make sure each probe is actually in the training data (may not be if isolated in test data)
    structure2probe2cat = defaultdict(dict)
    structures = [params.corpus]
    for structure in structures:
        probe2cat = load_probe2cat(project_path, structure)
        for probe, cat in probe2cat.items():
            num_in_train = prep.tokens_train.count(probe)
            num_in_valid = prep.tokens_valid.count(probe)
            if num_in_train == 0:
                if num_in_valid == 0:
                    raise RuntimeError(f'"{probe:<24}" not in train or test data after tokenization.')

            else:
                structure2probe2cat[structure][probe] = cat

    # model
    model = RNN(
        params.flavor,
        prep.num_types,
        params.hidden_size,
        params.num_layers,
    )

    # loss function
    criterion = torch.nn.CrossEntropyLoss()
    if params.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=params.lr)
    elif params.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=params.lr)
    else:
        raise AttributeError('Invalid arg to "optimizer"')

    # initialize dictionary for collecting performance data
    performance = {'train_pp': [], 'test_pp': []}

    # train and eval
    eval_steps = []  # to keep track when performance is evaluated
    start_train = time.time()
    pbar = pyprind.ProgBar(num_train_mbs, stream=1)
    for step, windows in enumerate(batch_generator):

        if step != 0:
            context_size = windows.shape[1] - 1  # different depending on whether input is from prep_start
            x, y = np.split(windows, [context_size], axis=1)
            inputs = torch.cuda.LongTensor(x)
            targets = torch.cuda.LongTensor(np.squeeze(y))

            # forward step
            model.batch_size = len(windows)  # dynamic batch size
            model.train()
            logits = model(inputs)['logits']  # initial hidden state defaults to zero if not provided

            # backward step
            optimizer.zero_grad()  # sets all gradients to zero
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        pbar.update()

        # evaluate performance
        if step % configs.Eval.num_steps_to_eval == 0 \
                or step in high_resolution_eval_steps:  # eval with higher resolution at start
            eval_steps.append(step)
            model.eval()
            performance = update_pp_performance(performance, model, criterion, prep)

            performance = update_ba_performance(performance, model, prep, structure2probe2cat)

            for k, v in performance.items():
                if not v:
                    continue
                print(f'{k: <12}={v[-1]:.2f}')
            print(flush=True)

            # print progress to console
            minutes_elapsed = int(float(time.time() - start_train) / 60)
            print(f'completed step={step:>12,}/{num_train_mbs:>12,}')
            print(f'minutes elapsed={minutes_elapsed}')
            print(flush=True)

    # collect performance in list of pandas series
    res = []
    for k, v in performance.items():
        if not v:
            continue
        transcript = pd.Series(v, index=eval_steps)
        transcript.name = k
        res.append(transcript)

    return res
