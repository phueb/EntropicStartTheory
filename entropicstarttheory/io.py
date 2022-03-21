from typing import Dict
from pathlib import Path


def load_probe2cat(project_path: Path,
                   corpus_name: str,
                   ) -> Dict[str, str]:
    res = {}
    num_total = 0
    path_probes = project_path / 'data' / 'structures' / f'semantic_categories_{corpus_name}.txt'
    with path_probes.open('r') as f:
        for line in f:
            data = line.strip().strip('\n').split()
            probe = data[1]
            cat = data[0]

            res[probe] = cat
            num_total += 1

    return res
