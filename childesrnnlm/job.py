import time
import pyprind
import pandas as pd
import numpy as np
import torch
from collections import defaultdict, Counter
from pathlib import Path
from itertools import chain, product
from typing import List
import random

from aochildes.dataset import ChildesDataSet
from aonewsela.dataset import NewselaDataSet
from preppy import Prep
from entropicstart.editor import Editor

from childesrnnlm import configs
from childesrnnlm.bpe import train_bpe_tokenizer
from childesrnnlm.io import load_probe2cat
from childesrnnlm.evaluation import calc_perplexity
from childesrnnlm.evaluation import eval_ba_performance
from childesrnnlm.evaluation import eval_si_performance
from childesrnnlm.evaluation import eval_sd_performance
from childesrnnlm.evaluation import eval_pr1_performance
from childesrnnlm.evaluation import eval_pr2_performance
from childesrnnlm.evaluation import eval_ma_performance
from childesrnnlm.evaluation import eval_pd_performance
from childesrnnlm.evaluation import eval_cs_performance
from childesrnnlm.evaluation import eval_cc_performance
from childesrnnlm.evaluation import eval_op_performance
from childesrnnlm.evaluation import eval_en_performance
from childesrnnlm.evaluation import eval_eo_performance
from childesrnnlm.evaluation import eval_fr_performance
from childesrnnlm.evaluation import get_context2f
from childesrnnlm.representation import make_inp_representations, make_out_representations
from childesrnnlm.params import Params
from childesrnnlm.rnn import RNN


def main(param2val):
    # params
    params = Params.from_param2val(param2val)
    print(params)

    project_path = Path(param2val['project_path'])
    save_path = Path(param2val['save_path'])

    # in case, job is run locally, we must create save_path
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # load corpus
    if params.corpus == 'aochildes':
        transcripts = ChildesDataSet().load_transcripts()
    elif params.corpus == 'aonewsela':
        transcripts = NewselaDataSet().load_transcripts()
    else:
        raise AttributeError('Invalid corpus')

    # shuffle at transcript level
    if params.shuffle_transcripts:
        random.shuffle(transcripts)

    if params.num_transcripts is not None:  # for debugging
        print(f'WARNING: Using {params.num_transcripts} transcripts')
        transcripts = transcripts[:params.num_transcripts]

    text_original = ' '.join(transcripts)
    tokens_original = text_original.split()
    print(f'Loaded {len(tokens_original):,} words.')

    # probes should never be split by tokenizer.
    # so, we replace all probes by a single special token.
    # using only 1 special token speeds tokenization - but we must re-populate probes after tokenization
    special_token = '<probe>'

    # get probes for evaluation, and check they are in the corpus
    probes_in_data = set()
    num_total = 0
    types_in_sentences = set(tokens_original)
    for structure in configs.Eval.structures:
        probe2cat = load_probe2cat(project_path, structure, params.corpus)
        num_total += len(probe2cat)
        for probe in probe2cat.keys():
            if probe in types_in_sentences:
                probes_in_data.add(probe)
            else:
                print(f'probe={probe:<24} not in original data. Excluded.')
        print(f'structure={structure:<24} | {len(probes_in_data)} of {num_total} total probes occur in original data')
    probes = list(probes_in_data)
    for probe in probes:
        assert probe in text_original

    # replace probes in corpus with special_token
    tokens_special = []
    replaced_probes = []
    for token in tokens_original:
        if token in probes_in_data:
            tokens_special.append(special_token)
            replaced_probes.append(token)
        else:
            tokens_special.append(token)

    # tokenize text
    print(f'Tokenizing {len(transcripts)} transcripts..', flush=True)
    tokenizer = train_bpe_tokenizer(transcripts, params.num_types, special_tokens=[special_token])
    text_special = ' '.join(tokens_special)
    tokens: List[str] = [t for t in tokenizer.encode(text_special, add_special_tokens=True).tokens
                         if t not in {'Ġ', '', ' '}]
    print(f'{len(set(tokens)):,} types in tokenized text', flush=True)
    print(f'Tokenized text has {len(tokens) - len(tokens_original):,} more tokens than before tokenization')

    # replace special token with original probes
    replaced_probes_it = iter(replaced_probes)
    tokens = [token if token != special_token else next(replaced_probes_it) for token in tokens]
    assert len(list(replaced_probes_it)) == 0

    # check that added tokens were not split during tokenization
    num_errors = 0
    for special_t in probes:
        if special_t not in tokens and special_t in tokens_original:
            print(f'{special_t:<24} occurs {tokens_original.count(special_t)} times in original text '
                  f'but not in tokenized text.')
            num_errors += 1
    if num_errors:
        raise RuntimeError(f'{num_errors} special tokens were not found in tokenized text.')

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
        print(f'Adding {params.start} start', flush=True)
        editor = Editor(tokens, probes, num_parts=params.num_parts)
        tokens_start = editor.make_start_tokens(params.start,
                                                num_left_words=configs.Start.num_left_words,
                                                num_right_words=configs.Start.num_right_words)
        prep_start = Prep(tokens_start,
                          reverse=False,
                          sliding=False,
                          num_parts=1,
                          num_iterations=params.num_iterations,
                          batch_size=params.batch_size,
                          context_size=2,
                          token2id=prep.token2id
                          )
        assert prep_start.token2id == prep.token2id
        print(f'First {prep_start.num_mbs} batches are reserved for start sentences')
    else:
        prep_start = None
        print(f'Not adding start.')

    # combine start sequences and regular sequences
    if prep_start:
        batch_generator = chain(prep_start.generate_batches(), prep.generate_batches())
        high_resolution_eval_steps = list(range(0, prep_start.num_mbs, prep_start.num_mbs // 10))
        num_train_mbs = prep_start.num_mbs + prep.num_mbs
    else:
        batch_generator = prep.generate_batches()
        high_resolution_eval_steps = configs.Eval.high_res_eval_steps
        num_train_mbs = prep.num_mbs

    # load all structures, for evaluation, each consisting of a dict mapping probe -> category,
    # make sure each probe is actually in the training data (may not be if isolated in test data)
    print(f'Checking that probes are in tokenized data...', flush=True)
    counter_train = Counter(prep.tokens_train)
    counter_test_ = Counter(prep.tokens_test_)
    structure2probe2cat = defaultdict(dict)
    for structure in configs.Eval.structures:
        probe2cat = load_probe2cat(project_path, structure, params.corpus)
        for probe, cat in probe2cat.items():
            if probe not in probes_in_data:
                continue
            num_in_train = counter_train[probe]
            num_in_test_ = counter_test_[probe]
            if num_in_train == 0:
                if num_in_test_ == 0:
                    if params.num_transcripts is None:  # do not raise exception when debugging
                        raise RuntimeError(f'"{probe:<24}" not in train or test data after tokenization.')

            else:
                structure2probe2cat[structure][probe] = cat

    # TODO why is prep.num_types > params.num_types?
    #  it's probably because tokenizer is trained to split probes into sub-words,
    #  and the probes are added as whole-words later into prep

    # model
    model = RNN(
        params.flavor,
        prep.num_types,
        params.hidden_size,
        params.num_layers,
        params.bias,
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
            inputs = torch.LongTensor(x).cuda()
            targets = torch.LongTensor(np.squeeze(y)).cuda()

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

            # evaluate perplexity
            if configs.Eval.train_pp:
                train_pp = calc_perplexity(model, criterion, prep, is_test=False)
                performance['train_pp'].append(train_pp)
            if configs.Eval.min_num_test_tokens > 0:
                test_pp = calc_perplexity(model, criterion, prep, is_test=True)
                performance['test_pp'].append(test_pp)

            # evaluate the semantic space using probe words
            for structure_name in configs.Eval.structures:

                # get probes
                probe2cat = structure2probe2cat[structure_name]
                probes = sorted(probe2cat.keys())

                # save probe representations to shared drive (for offline clustering analysis)
                probe_reps_inp = make_inp_representations(model, probes, prep, 'n')
                probe_reps_out = make_out_representations(model, probes, prep, 'n')
                np.savez_compressed(save_path / f'probe_reps_{step:0>12}',
                                    probe_reps_inp=probe_reps_inp,
                                    probe_reps_out=probe_reps_out)

                for direction, location, context_type in product(
                        configs.Eval.directions,
                        configs.Eval.locations,
                        configs.Eval.context_types,
                ):

                    performance_name = '{}_' + f'{structure_name}_{direction}_{location}_{context_type}'

                    # get words for evaluation
                    if direction in {'l', 'r'}:
                        type_eval2f = get_context2f(prep, probes, direction)
                        types_eval = list(type_eval2f.keys())
                    elif direction == 'c':
                        type_eval2f = None  # we do not weight probes by their frequency
                        types_eval = probes
                    else:
                        raise AttributeError('Invalid arg for direction.')

                    print(f'Found {len(types_eval):,} types for evaluation', flush=True)
                    print(f'Evaluating representations '
                          f'with direction={direction} location={location} context_type={context_type}', flush=True)

                    # make representations
                    if location == 'out':
                        representations = make_out_representations(model, types_eval, prep, context_type)
                    elif location == 'inp':
                        representations = make_inp_representations(model, types_eval, prep, context_type)
                    else:
                        raise AttributeError('Invalid arg to location')

                    assert len(representations) > 0
                    assert np.ndim(representations) == 2

                    if configs.Eval.calc_ba and direction == 'c':
                        print('Computing balanced accuracy...', flush=True)
                        start_eval = time.time()
                        res = eval_ba_performance(representations, probe2cat)
                        performance.setdefault(performance_name.format('ba'), []).append(res)
                        print(f'Elapsed={time.time() - start_eval}secs', flush=True)

                    if configs.Eval.calc_si and direction == 'c':
                        print('Computing silhouette score...', flush=True)
                        start_eval = time.time()
                        res = eval_si_performance(representations, probe2cat)
                        performance.setdefault(performance_name.format('si'), []).append(res)
                        print(f'Elapsed={time.time() - start_eval}secs', flush=True)

                    if configs.Eval.calc_sd and direction == 'c':
                        print('Computing S_Dbw...', flush=True)
                        start_eval = time.time()
                        res = eval_sd_performance(representations, probe2cat)
                        performance.setdefault(performance_name.format('sd'), []).append(res)
                        print(f'Elapsed={time.time() - start_eval}secs', flush=True)

                    if configs.Eval.calc_ma:
                        print('Computing magnitude...', flush=True)
                        start_eval = time.time()
                        res = eval_ma_performance(representations)
                        performance.setdefault(performance_name.format('ma'), []).append(res)
                        print(f'Elapsed={time.time() - start_eval}secs', flush=True)

                    if configs.Eval.calc_pr1 and location == 'inp' and context_type == 'n':
                        print('Computing actual-prototype to theoretical-prototype distance...', flush=True)
                        start_eval = time.time()
                        res = eval_pr1_performance(representations, types_eval, prep, model)
                        performance.setdefault(performance_name.format('pr1'), []).append(res)
                        print(f'Elapsed={time.time() - start_eval}secs', flush=True)

                    if configs.Eval.calc_pr2 and location == 'out' and context_type == 'n':
                        print('Computing exemplar to theoretical-prototype distance...', flush=True)
                        start_eval = time.time()
                        res = eval_pr2_performance(representations, types_eval, prep)
                        performance.setdefault(performance_name.format('pr2'), []).append(res)
                        print(f'Elapsed={time.time() - start_eval}secs', flush=True)

                    if configs.Eval.calc_pd and location == 'out':
                        print('Computing pairwise divergences...', flush=True)
                        start_eval = time.time()
                        res = eval_pd_performance(representations)
                        performance.setdefault(performance_name.format('pd'), []).append(res)
                        print(f'Elapsed={time.time() - start_eval}secs', flush=True)

                    if configs.Eval.calc_cs:
                        print('Computing cosine similarity...', flush=True)
                        start_eval = time.time()
                        res = eval_cs_performance(representations)
                        performance.setdefault(performance_name.format('cs'), []).append(res)
                        print(f'Elapsed={time.time() - start_eval}secs', flush=True)

                    if configs.Eval.calc_cc:
                        print('Computing cosine similarity within each category...', flush=True)
                        start_eval = time.time()
                        res = eval_cc_performance(representations, probe2cat)
                        performance.setdefault(performance_name.format('cc'), []).append(res)
                        print(f'Elapsed={time.time() - start_eval}secs', flush=True)

                    if configs.Eval.calc_op and location == 'inp':
                        print('Computing divergence between origin and prototype...', flush=True)
                        start_eval = time.time()
                        res = eval_op_performance(model, prep, types_eval)
                        performance.setdefault(performance_name.format('op'), []).append(res)
                        print(f'Elapsed={time.time() - start_eval}secs', flush=True)

                    if configs.Eval.calc_en and location == 'out':
                        print('Computing entropy of representations...', flush=True)
                        start_eval = time.time()
                        res = eval_en_performance(representations)
                        performance.setdefault(performance_name.format('en'), []).append(res)
                        print(f'Elapsed={time.time() - start_eval}secs', flush=True)

                    if configs.Eval.calc_eo and location == 'inp':
                        print('Computing entropy of representations of origin...', flush=True)
                        start_eval = time.time()
                        res = eval_eo_performance(representations)
                        performance.setdefault(performance_name.format('eo'), []).append(res)
                        print(f'Elapsed={time.time() - start_eval}secs', flush=True)

                    if configs.Eval.calc_fr:
                        print('Computing fragmentation and condition-number...', flush=True)
                        start_eval = time.time()
                        fr, co = eval_fr_performance(representations, type_eval2f)
                        performance.setdefault(performance_name.format('fr'), []).append(fr)
                        performance.setdefault(performance_name.format('co'), []).append(co)
                        print(f'Elapsed={time.time() - start_eval}secs', flush=True)

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
