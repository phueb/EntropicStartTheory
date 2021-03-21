from typing import List, Dict

from childesrnnlm import configs


def load_probe2cat(probes_name: str,
                   corpus_name: str,
                   ) -> Dict[str, str]:
    res = {}
    num_total = 0
    path_probes = configs.Dirs.structures / corpus_name / f'{probes_name}.txt'
    with path_probes.open('r') as f:
        for line in f:
            data = line.strip().strip('\n').split()
            probe = data[0]
            cat = data[1]

            res[probe] = cat
            num_total += 1

    return res
