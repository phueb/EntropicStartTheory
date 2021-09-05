import random
from typing import Dict, Optional, List


artificial_corpus_names = {'axy',  # semantic category signal is in right neighbor,
                           'yxb',  # semantic category signal is in left neighbor,
                           'rxy',  # semantic category signal is in right neighbor + lexically redundant left neighbor,
                           'yxy',  # semantic category signal is in right neighbor + left neighbor,
                           }


class AXBDataSet:
    """
    make documents containing strings with following the structure (A, X, B, .).

    used to test if RNN learns semantic category information (Y) faster when it is before or after a target word (X)

    """

    def __init__(self,
                 dependency_type: str,
                 probe2cat: Dict[str, str],
                 num_docs: int = 1,
                 num_tokens_per_doc: int = 200_000,
                 seed: Optional[int] = None,
                 ) -> None:

        if dependency_type not in artificial_corpus_names:
            raise AttributeError('Invalid arg to dependency_type')

        self.dependency_type = dependency_type
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
        self.y = ['.']

        self.types = self.a + self.b + self.x + self.y  # order alphabetically

        # map xi to category-relevant a and b subset
        a_subsets = [self.a[offset::self.num_categories] for offset in range(self.num_categories)]
        b_subsets = [self.b[offset::self.num_categories] for offset in range(self.num_categories)]
        xi2cat_id = {xi: categories.index(cat) for xi, cat in probe2cat.items()}
        self.xi2a_fragment = {xi: a_subsets[xi2cat_id[xi]] for xi in self.x}
        self.xi2b_fragment = {xi: b_subsets[xi2cat_id[xi]] for xi in self.x}

        # make each xi redundant with one ai, bi
        self.xi2ai = {xi: ai for xi, ai in zip(self.x, self.a)}
        self.xi2bi = {xi: bi for xi, bi in zip(self.x, self.b)}

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

            # make ai or bi semantically related to xi
            if self.dependency_type in {'axy', 'rxy', 'yxy'}:
                bi = random.choice(self.xi2b_fragment[xi])
                if self.dependency_type == 'rxy':  # in addition, we create redundancy at left neighbor
                    ai = self.xi2ai[xi]
                elif self.dependency_type == 'yxy':
                    ai = random.choice(self.xi2a_fragment[xi])
            elif self.dependency_type == 'yxb':
                ai = random.choice(self.xi2a_fragment[xi])
            else:
                raise AttributeError('Invalid arg to dependency_type')

            # collect
            res += f'{ai} {xi} {bi} . '  # whitespace after each

        return res
