import time
import pyprind
import attr
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from itertools import product

from preppy.latest import Prep

# from categoryeval.score import Scorer
from prototypeeval.score import Scorer

from startingabstract import config
from startingabstract.docs import load_docs
from startingabstract.evaluation import update_ba_metrics, update_pp_metrics, update_dp_metrics
from startingabstract.rnn import RNN


@attr.s
class Params(object):
    reverse = attr.ib(validator=attr.validators.instance_of(bool))
    shuffle_docs = attr.ib(validator=attr.validators.instance_of(bool))
    corpus = attr.ib(validator=attr.validators.instance_of(str))
    ba_names = attr.ib(validator=attr.validators.instance_of(list))
    dp_names = attr.ib(validator=attr.validators.instance_of(list))
    num_types = attr.ib(validator=attr.validators.instance_of(int))
    slide_size = attr.ib(validator=attr.validators.instance_of(int))
    context_size = attr.ib(validator=attr.validators.instance_of(int))
    batch_size = attr.ib(validator=attr.validators.instance_of(int))
    flavor = attr.ib(validator=attr.validators.instance_of(str))
    hidden_size = attr.ib(validator=attr.validators.instance_of(int))
    lr = attr.ib(validator=attr.validators.instance_of(float))
    optimizer = attr.ib(validator=attr.validators.instance_of(str))

    @classmethod
    def from_param2val(cls, param2val):
        kwargs = {k: v for k, v in param2val.items()
                  if k not in ['job_name', 'param_name', 'save_path', 'project_path']}
        return cls(**kwargs)


def main(param2val):

    # params
    params = Params.from_param2val(param2val)
    print(params)

    project_path = Path(param2val['project_path'])
    corpus_path = project_path / 'data' / f'{params.corpus}.txt'
    train_docs, test_docs = load_docs(corpus_path,
                                      params.shuffle_docs)

    # prepare input
    train_prep = Prep(train_docs,
                      params.reverse,
                      params.num_types,
                      params.slide_size,
                      params.batch_size,
                      params.context_size,
                      config.Eval.num_evaluations,
                      )

    # TODO does test prep need to be different other than vocab?

    test_slide_size = params.batch_size
    test_prep = Prep(test_docs,
                     params.reverse,
                     params.num_types,
                     test_slide_size,  # TODO set slide_size to batch-size
                     params.batch_size,
                     params.context_size,
                     config.Eval.num_evaluations,
                     vocab=train_prep.store.types
                     )
    windows_generator = train_prep.gen_windows()  # has to be created once

    gen_size = len([1 for i in train_prep.gen_windows()])
    print(f'Number of total batches={gen_size}')

    # classes that perform scoring
    # TODO initialize a ba_scorer here that has multiple probe stores as properties

    dp_scorer = Scorer(train_prep.store.tokens, params.dp_names, config.Eval.dp_num_parts)

    # model
    model = RNN(
        params.flavor,
        params.num_types,
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

    # initialize metrics for evaluation
    metrics = {
        'train_pp': [],
        'test_pp': [],
        config.Metrics.ba_o: [],
        config.Metrics.ba_n: [],
    }
    for dp_name, part in product(params.dp_names, range(config.Eval.dp_num_parts)):
        metrics[f'dp_{dp_name}_part{part}'] = []

    # train and eval
    train_mb = 0
    start_train = time.time()
    for timepoint, data_mb in enumerate(train_prep.eval_mbs):

        # train
        if timepoint != 0:
            train_mb = train_on_corpus(model, optimizer, criterion, train_prep, data_mb, train_mb, windows_generator)

        # eval (metrics must be returned to reuse the same object)
        # metrics = update_dp_metrics(metrics, model, train_prep, dp_scorer)
        # metrics = update_pp_metrics(metrics, model, criterion, train_prep, test_prep)
        # metrics = update_ba_metrics(metrics, model, train_prep, probe_store)

        # print progress to console
        minutes_elapsed = int(float(time.time() - start_train) / 60)
        print(f'completed time-point={timepoint} of {config.Eval.num_evaluations}')
        print(f'minutes elapsed={minutes_elapsed}')
        for k, v in metrics.items():
            if not v:
                continue
            print(f'{k: <12}={v[-1]:.2f}')
        print(flush=True)

    # collect performance in list of pandas series
    res = []
    for k, v in metrics.items():
        if not v:
            continue
        s = pd.Series(v, index=train_prep.eval_mbs)
        s.name = k
        res.append(s)

    return res


def train_on_corpus(model, optimizer, criterion, prep, data_mb, train_mb, windows_generator):
    print('Training on items from mb {:,} to mb {:,}...'.format(train_mb, data_mb))
    pbar = pyprind.ProgBar(data_mb - train_mb, stream=1)
    model.train()
    for windows in windows_generator:

        x, y = np.split(windows, [prep.context_size], axis=1)
        inputs = torch.cuda.LongTensor(x)
        targets = torch.cuda.LongTensor(np.squeeze(y))

        # forward step
        model.batch_size = len(windows)  # dynamic batch size
        logits = model(inputs)['logits']  # initial hidden state defaults to zero if not provided

        # backward step
        optimizer.zero_grad()  # sets all gradients to zero
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        pbar.update()

        train_mb += 1  # has to be like this, because enumerate() resets
        if data_mb == train_mb:
            return train_mb