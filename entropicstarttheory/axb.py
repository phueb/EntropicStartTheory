import random
from typing import Dict, Optional, List
import re
from dataclasses import dataclass


artificial_corpus_structures = {'axy',  # semantic category signal is in right neighbor,
                                'yxb',  # semantic category signal is in left neighbor,
                                'rxy',  # semantic category signal is in right neighbor + lexical redundant left,
                                'yxy',  # semantic category signal is in right neighbor AND or OR left neighbor,
                                }


@dataclass
class AXBParams:
    rule: Optional[str] = None
    redundancy: Optional[float] = None
    prop_fully_determined: Optional[float] = None  # the proportion of Y types that are perfectly determined by A
    redundant_with: Optional[str] = None


class AXBDataSet:
    """
    make documents containing strings with following the structure (A, X, B, .).

    used to test if RNN learns semantic category information (Y) faster when it is before or after a target word (X)

    """

    def __init__(self,
                 corpus_name: str,
                 probe2cat: Dict[str, str],
                 num_docs: int = 1,
                 num_tokens_per_doc: int = 400_000,
                 seed: Optional[int] = None,
                 ) -> None:

        self.corpus_structure = corpus_name.split('-')[0]
        if self.corpus_structure not in artificial_corpus_structures:
            raise AttributeError(f'Did not recognize corpus structure "{self.corpus_structure}".')

        # parse corpus name to get rules for generating corpus
        params_kwargs = {}
        for substring in corpus_name.split('-')[1:]:
            pattern = re.compile(r'(?P<key>\w+):(?P<value>.+)')  # 2 groups, called "key" and "value"
            match = pattern.match(substring)
            k = match.group('key')
            v = match.group('value')
            params_kwargs[k] = v
        self.axb_params = AXBParams(**params_kwargs)

        if self.axb_params.prop_fully_determined is not None:
            if self.axb_params.redundancy is not None:
                raise AttributeError('When "prop_y_determined_by_a" is not None, "redundancy" must be None.')

        print(f'Initializing AXB corpus with:\n{self.axb_params}')

        self.num_docs = num_docs
        self.num_tokens_per_doc = num_tokens_per_doc

        self.num_tokens_in_window = 4
        self.slots = ['a', 'x', 'b', '.']
        self.num_windows_per_doc = self.num_tokens_per_doc // self.num_tokens_in_window

        self.x = [probe for probe, cat in probe2cat.items()]
        categories = sorted(set(probe2cat.values()))
        self.num_categories = len(categories)

        self.num_a = len(self.x)  # this is required so that each xi can have a redundant ai
        self.num_x = len(self.x)
        self.num_b = len(self.x)  # this is required so that each xi can have a redundant bi
        self.num_types = self.num_a + self.num_x + self.num_b + 1

        self.a = [f'{self.slots[0]}{i:0>6}' for i in range(self.num_a)]
        self.b = [f'{self.slots[2]}{i:0>6}' for i in range(self.num_b)]

        # map xi to category-relevant a and b subset
        a_subsets = [self.a[offset::self.num_categories] for offset in range(self.num_categories)]
        b_subsets = [self.b[offset::self.num_categories] for offset in range(self.num_categories)]
        xi2cat_id = {xi: categories.index(cat) for xi, cat in probe2cat.items()}
        self.xi2a_fragment = {xi: a_subsets[xi2cat_id[xi]] for xi in self.x}
        self.xi2b_fragment = {xi: b_subsets[xi2cat_id[xi]] for xi in self.x}

        # make each xi redundant with one ai, bi
        self.xi2ai = {xi: ai for xi, ai in zip(self.x, self.a)}
        self.xi2bi = {xi: bi for xi, bi in zip(self.x, self.b)}

        # make each ai redundant with one bi and vice versa
        self.ai2bi = {ai: bi for ai, bi in zip(self.a, self.b)}
        self.bi2ai = {bi: ai for ai, bi in zip(self.a, self.b)}

        # make some proportion of word types always perfectly predicted by another word type
        self.fully_determined_a = {}
        self.fully_determined_x = {}
        self.fully_determined_b = {}
        if self.axb_params.prop_fully_determined is not None:
            if self.axb_params.redundant_with == 'x':
                self.fully_determined_x = {xi for xi in self.x
                                           if random.random() < float(self.axb_params.prop_fully_determined)}
            elif self.axb_params.redundant_with == 'y':
                if self.corpus_structure == 'yxy':
                    raise NotImplementedError
                if self.corpus_structure[0] == 'y':
                    self.fully_determined_a = {ai for ai in self.a
                                               if random.random() < float(self.axb_params.prop_fully_determined)}
                elif self.corpus_structure[2] == 'y':
                    self.fully_determined_b = {bi for bi in self.b
                                               if random.random() < float(self.axb_params.prop_fully_determined)}
            else:
                raise AttributeError('redundant_with must be "x" or "y".')

        if seed is not None:
            random.seed(seed)

    def load_documents(self) -> List[str]:

        res = []
        for doc_id in range(self.num_docs):
            doc = self.make_doc()
            res.append(doc)

        return res

    def make_doc(self,
                 ) -> str:

        res = ''
        for n in range(self.num_windows_per_doc):

            # sample randomly
            xi = random.choice(self.x)
            ai = random.choice(self.a)
            bi = random.choice(self.b)

            # make ai and/or bi semantically related to xi
            if self.corpus_structure == 'axy':
                ai = ai
                bi = random.choice(self.xi2b_fragment[xi])

            elif self.corpus_structure == 'yxb':
                ai = random.choice(self.xi2a_fragment[xi])
                bi = bi

            elif self.corpus_structure == 'yxy':
                if self.axb_params.rule == 'or':
                    modify_a = random.choice([True, False])
                    modify_b = not modify_a
                elif self.axb_params.rule == 'and':
                    modify_a = True
                    modify_b = True
                else:
                    raise AttributeError('Invalid arg to "rule".')
                if modify_a:
                    ai = random.choice(self.xi2a_fragment[xi])
                if modify_b:
                    bi = random.choice(self.xi2b_fragment[xi])

            elif self.corpus_structure == 'rxy':
                bi = random.choice(self.xi2b_fragment[xi])
                # probabilistic redundancy
                if self.axb_params.redundancy is not None:
                    if random.random() < float(self.axb_params.redundancy):
                        if self.axb_params.redundant_with == 'x':  # ai is lexically redundant with xi
                            ai = self.xi2ai[xi]
                        elif self.axb_params.redundant_with == 'y':  # ai is lexically redundant with bi
                            ai = self.bi2ai[bi]
                        else:
                            raise AttributeError('redundant_with must be "x" or "y".')
                # deterministic redundancy
                elif xi in self.fully_determined_x:  # ai is lexically redundant with xi
                    ai = self.xi2ai[xi]
                elif bi in self.fully_determined_b:  # ai is lexically redundant with bi
                    ai = self.bi2ai[bi]

            else:
                raise AttributeError('Invalid arg to "corpus_structure".')

            # collect
            res += f'{ai} {xi} {bi} . '  # whitespace after each

        return res
