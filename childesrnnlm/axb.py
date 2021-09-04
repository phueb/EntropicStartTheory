import random
from typing import Dict, Optional, List


class AXBDataSet:
    """
    make documents containing strings with following the structure (A, X, B, .).

    used to test if RNN learns semantic category information (Y) faster when it is before or after a target word (X)

    """

    def __init__(self,
                 dependency_type: str,
                 probe2cat: Dict[str, str],
                 num_docs: int = 1,
                 num_tokens_per_doc: int = 100_000,
                 num_a_per_category: int = 10,
                 num_b_per_category: int = 10,
                 seed: Optional[int] = None,
                 ) -> None:

        if dependency_type not in {'axy', 'yxb'}:
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

        self.num_a = num_a_per_category * self.num_categories
        self.num_x = len(self.x)
        self.num_b = num_b_per_category * self.num_categories
        self.num_types = num_a_per_category + + self.num_x + num_b_per_category + 1

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
            if self.dependency_type == 'axy':
                bi = random.choice(self.xi2b_fragment[xi])
            elif self.dependency_type == 'yxb':
                ai = random.choice(self.xi2a_fragment[xi])
            else:
                raise AttributeError('Invalid arg to dependency_type')

            # collect
            res += f'{ai} {xi} {bi} . '  # whitespace after each

        return res
