import time
import pyprind
import attr
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from itertools import islice
from itertools import product
from typing import Iterator

from preppy.latest import Prep
from preppy.legacy import TrainPrep, TestPrep

from categoryeval.ba import BAScorer
from categoryeval.dp import DPScorer

from startingabstract import config
from startingabstract.docs import load_docs
from startingabstract.evaluation import update_ba_metrics
from startingabstract.evaluation import update_pp_metrics
from startingabstract.evaluation import update_dp_metrics_conditional
from startingabstract.evaluation import update_dp_metrics_unconditional
from startingabstract.rnn import RNN


@attr.s
class Params(object):
    legacy = attr.ib(validator=attr.validators.instance_of(bool))
    reverse = attr.ib(validator=attr.validators.instance_of(bool))
    shuffle_sentences = attr.ib(validator=attr.validators.instance_of(bool))
    corpus = attr.ib(validator=attr.validators.instance_of(str))
    ba_probes = attr.ib(validator=attr.validators.instance_of(tuple))
    dp_probes = attr.ib(validator=attr.validators.instance_of(tuple))
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
    corpus_path = project_path / 'corpora' / f'{params.corpus}.txt'
    train_docs, test_docs = load_docs(corpus_path,
                                      shuffle_sentences=params.shuffle_sentences,
                                      num_test_docs=config.Eval.num_test_docs,
                                      )

    # prepare input
    if params.legacy:
        print('=======================================================')
        print('WARNING: params.legacy=True')
        print('=======================================================')

        num_iterations = 16
        assert config.Eval.num_total_ticks % num_iterations == 0  # TODO is this really important?
        train_prep = TrainPrep(train_docs,
                               params.reverse,
                               params.num_types,

                               num_parts=128,  # TODO test different num_parts

                               num_iterations=[num_iterations, num_iterations],
                               batch_size=params.batch_size,
                               context_size=params.context_size,
                               num_evaluations=config.Eval.num_total_ticks,
                               shuffle_within_part=False)
        test_prep = TestPrep(test_docs,
                             batch_size=params.batch_size,
                             context_size=params.context_size,
                             vocab=train_prep.store.types)
    else:
        train_prep = Prep(train_docs,
                          reverse=params.reverse,
                          num_types=params.num_types,
                          slide_size=params.slide_size,
                          batch_size=params.batch_size,
                          context_size=params.context_size,
                          num_evaluations=config.Eval.num_total_ticks)
        test_prep = Prep(test_docs,
                         reverse=params.reverse,
                         num_types=params.num_types,
                         slide_size=params.batch_size,
                         batch_size=params.batch_size,
                         context_size=params.context_size,
                         num_evaluations=config.Eval.num_total_ticks,
                         vocab=train_prep.store.types)

    windows_generator = train_prep.gen_windows()  # has to be created once

    # classes that perform scoring
    ba_scorer = BAScorer(params.corpus,
                         params.ba_probes,
                         train_prep.store.w2id
                         )
    dp_scorer = DPScorer(params.corpus,
                         params.dp_probes,
                         train_prep.store.tokens,
                         train_prep.store.types,
                         config.Eval.dp_num_parts
                         )

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
    metrics = {'train_pp': [], 'test_pp': []}
    for probes_name in params.ba_probes:
        metrics[f'ba_o_{probes_name}'] = []
        metrics[f'ba_n_{probes_name}'] = []
    for probes_name, part in product(params.dp_probes, range(config.Eval.dp_num_parts)):
        metrics[f'dp_{probes_name}_part{part}_js'] = []
        metrics[f'dp_{probes_name}_part{part}_xe'] = []
        metrics[f'dp_{probes_name}_part{part}_js_unconditional'] = []
        metrics[f'dp_{probes_name}_part{part}_xe_unconditional'] = []

    # train and eval
    train_mb = 0
    performance_mbs = []  # to keep track when performance is evaluated
    start_train = time.time()
    for tick, eval_mb in enumerate(train_prep.eval_mbs):
        # train
        if tick != 0:
            model.train()
            print(f'Training on items from mb {train_mb:,} to mb {eval_mb:,}...')
            train_on_corpus(model, optimizer, criterion, train_prep, windows_generator)
            train_mb += train_prep.num_mbs_per_eval

        # evaluate performance more frequently at start of training
        if tick < config.Eval.num_start_ticks or tick % config.Eval.tick_step == 0:
            model.eval()
            metrics = update_dp_metrics_conditional(metrics, model, train_prep, dp_scorer)
            metrics = update_dp_metrics_unconditional(metrics, model, train_prep, dp_scorer)
            metrics = update_pp_metrics(metrics, model, criterion, train_prep, test_prep)  # TODO causing CUDA error?
            metrics = update_ba_metrics(metrics, model, train_prep, ba_scorer)

            for k, v in metrics.items():
                if not v:
                    continue
                print(f'{k: <12}={v[-1]:.2f}')
            print(flush=True)

        # print progress to console
        minutes_elapsed = int(float(time.time() - start_train) / 60)
        print(f'completed time-point={tick} of {config.Eval.num_total_ticks}')
        print(f'minutes elapsed={minutes_elapsed}')
        print(flush=True)

    # collect performance in list of pandas series
    res = []
    for k, v in metrics.items():
        if not v:
            continue
        s = pd.Series(v, index=performance_mbs)
        s.name = k
        res.append(s)

    return res


def train_on_corpus(model: RNN,
                    optimizer: torch.optim.Optimizer,
                    criterion: torch.nn.CrossEntropyLoss,
                    prep: Prep,
                    windows_generator: Iterator[np.ndarray],
                    ) -> None:
    pbar = pyprind.ProgBar(prep.num_mbs_per_eval, stream=1)
    for windows in islice(windows_generator, 0, prep.num_mbs_per_eval):

        x, y = np.split(windows, [prep.context_size], axis=1)
        inputs = torch.cuda.LongTensor(x)
        targets = torch.cuda.LongTensor(np.squeeze(y))  # TODO copying batch to GPU each time is costly

        # forward step
        model.batch_size = len(windows)  # dynamic batch size
        logits = model(inputs)['logits']  # initial hidden state defaults to zero if not provided

        # backward step
        optimizer.zero_grad()  # sets all gradients to zero
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        pbar.update()