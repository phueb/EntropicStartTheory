from sklearn.datasets import make_classification
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

NUM_PROBES = 700
NUM_CATEGORIES = 30
NUM_HIDDEN = 512

for class_sep in np.arange(0, 100, 1):
    x, y = make_classification(n_samples=NUM_PROBES,
                               n_features=NUM_HIDDEN,
                               n_informative=NUM_HIDDEN,
                               n_redundant=0,
                               n_repeated=0,
                               class_sep=class_sep,
                               n_clusters_per_class=1,
                               n_classes=NUM_CATEGORIES,
                               random_state=1,
                               )
    cosines = cosine_similarity(x)
    print(np.min(cosines), np.max(cosines))
#
#     res = np.rad2deg(np.arccos(np.clip(cosines, -1.0, 1.0))).mean()
#     print(res)

print()
print(np.std(np.array([-11, -10,  -9, -1, 1,  9, 10, 11]) * 10))
print(np.std(np.array([-10, -10, -10, -1, 1, 10, 10, 10]) * 10))
