"""
what happens to cosine similarity when the range on the unit sphere is contracted
so that vector must move closer to each other?

when range is contracted, cosine similarity goes up.
moreover, when range is contracted, it also becomes possible for clustering to increase cosine similarity also.
if the range is not contracted, clustering can NOT increase cosine similarity.

the conclusion is that, when cosine similarity goes up, it is impossible to know if this is due to :
1. only a contraction of the range on the unit sphere, or
2. both a contraction of the range, and additionally, clustering within that restricted range.

to specify no contraction, use CONTRACTION = 1 * np.pi
to specify no clustering, use CLUSTERING = 1

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


# define points on unit sphere which correspond to 4 clusters of each 2 NEARBY vectors
CONTRACTION = 0.67 * np.pi
CLUSTERING = 1
OFFSET = 0.25 / 2 * CONTRACTION / CLUSTERING
c1 = np.array([

    CONTRACTION * 0.0 - OFFSET,
    CONTRACTION * 0.0 + OFFSET,

    CONTRACTION * 0.5 - OFFSET,
    CONTRACTION * 0.5 + OFFSET,

    CONTRACTION * 1.0 - OFFSET,
    CONTRACTION * 1.0 + OFFSET,

    CONTRACTION * 1.5 - OFFSET,
    CONTRACTION * 1.5 + OFFSET,
    ])

# define points on unit sphere which correspond to 4 clusters of each 2 OVERLAPPING vectors
CONTRACTION = 0.67 * np.pi
CLUSTERING = 4
OFFSET = 0.25 / 2 * CONTRACTION / CLUSTERING
c2 = np.array([

    CONTRACTION * 0.0 - OFFSET,
    CONTRACTION * 0.0 + OFFSET,

    CONTRACTION * 0.5 - OFFSET,
    CONTRACTION * 0.5 + OFFSET,

    CONTRACTION * 1.0 - OFFSET,
    CONTRACTION * 1.0 + OFFSET,

    CONTRACTION * 1.5 - OFFSET,
    CONTRACTION * 1.5 + OFFSET,
    ])


for c in [c1, c2]:

    # make figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    # draw circle for reference
    t = np.linspace(0, 2 * np.pi, 360)
    ax.plot(np.cos(t), np.sin(t), ':', linewidth=1, color='grey')
    # draw points
    x = np.cos(c)
    y = np.sin(c)
    ax.scatter(x, y, linewidth=1)
    # show figure
    plt.show()

    # get matrix of coordinates (representing the vectors of interest)
    xy = np.hstack((x[:, np.newaxis], y[:, np.newaxis])) * 100
    assert xy.shape == (len(c), 2)

    # calc cosine
    sim = cosine_similarity(xy)

    # calc average over off-diagonals (ignore 1s)
    masked = np.ma.masked_where(np.eye(*sim.shape), sim)
    print(masked.mean())
    print(masked.std())
    print()
