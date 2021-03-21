import time
import pyprind
import pandas as pd
import numpy as np
import torch
from collections import defaultdict

from aochildes.dataset import ChildesDataSet

from preppy import FlexiblePrep

from childesrnnlm import configs
from childesrnnlm.io import load_probe2cat
from childesrnnlm.evaluation import update_ba_performance
from childesrnnlm.evaluation import update_pp_performance
from childesrnnlm.evaluation import update_dp_performance
from childesrnnlm.evaluation import update_cs_performance
from childesrnnlm.evaluation import update_si_performance
from childesrnnlm.evaluation import update_sd_performance
from childesrnnlm.params import Params
from childesrnnlm.rnn import RNN


def main(param2val):
    # params
    params = Params.from_param2val(param2val)
    print(params)

    # load childes data
    if params.corpus == 'aochildes':
        sentences = ChildesDataSet().load_sentences()[:params.num_sentences]
    elif params.corpus == 'newsela':
        raise NotImplementedError
    else:
        raise AttributeError('Invalid corpus')

    # TODO add option to reorder corpus based on entropy - import ordermatters

    # collect all probes, they should be treated as whole words by tokenizer
    probes_in_data = set()
    num_total = 0
    types_in_sentences = set(' '.join(sentences).split())
    for structure in configs.Eval.structures:
        probe2cat = load_probe2cat(structure, params.corpus)
        num_total += len(probe2cat)
        for probe in probe2cat.keys():
            if probe in types_in_sentences:
                probes_in_data.add(probe)
            else:
                print(f'"{probe:<24}" not in raw data. Excluded.')
        print(f'structure={structure:<24} | {len(probes_in_data)} of {num_total} total probes occur in raw data')

    # tokenize + vectorize text
    prep = FlexiblePrep(sentences,
                        reverse=params.reverse,
                        sliding=params.sliding,
                        num_types=params.num_types,
                        num_parts=params.num_parts,
                        num_iterations=params.num_iterations,
                        batch_size=params.batch_size,
                        context_size=params.context_size,
                        shuffle_within_part=False,
                        shuffle_sentences=params.shuffle_sentences,
                        min_num_test_tokens=configs.Eval.min_num_test_tokens,
                        special_tokens=list(probes_in_data),
                        )

    # load all structures, for evaluation, each consisting of a dict mapping probe -> category,
    # make sure each probe is actually in the training data (may not be if isolated in test data)
    structure2probe2cat = defaultdict(dict)
    for structure in configs.Eval.structures:
        probe2cat = load_probe2cat(structure, params.corpus)
        for probe, cat in probe2cat.items():
            if probe not in probes_in_data:
                continue

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
        prep.num_types,  # is larger than params.num_types due to added tokens
        params.hidden_size,
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
    pbar = pyprind.ProgBar(prep.num_mbs, stream=1)
    for step, windows in enumerate(prep.generate_batches()):

        if step != 0:
            x, y = np.split(windows, [prep.context_size], axis=1)
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
        if step % configs.Eval.num_steps_to_eval == 0:
            eval_steps.append(step)
            model.eval()
            performance = update_pp_performance(performance, model, criterion, prep)

            performance = update_ba_performance(performance, model, prep, structure2probe2cat)
            # performance = update_cs_performance(performance, model, prep, structure2probe2cat)
            # performance = update_dp_performance(performance, model, prep, structure2probe2cat)
            # performance = update_si_performance(performance, model, prep, structure2probe2cat)
            # performance = update_sd_performance(performance, model, prep, structure2probe2cat)

            for k, v in performance.items():
                if not v:
                    continue
                print(f'{k: <12}={v[-1]:.2f}')
            print(flush=True)

            # print progress to console
            minutes_elapsed = int(float(time.time() - start_train) / 60)
            print(f'completed step={step:>12,}/{prep.num_mbs:>12,}')
            print(f'minutes elapsed={minutes_elapsed}')
            print(flush=True)

    # collect performance in list of pandas series
    res = []
    for k, v in performance.items():
        if not v:
            continue
        s = pd.Series(v, index=eval_steps)
        s.name = k
        res.append(s)

    return res
