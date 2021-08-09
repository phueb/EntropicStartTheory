"""
when vectors on unit sphere cluster into equally spaced groups, does cosine similarity increase or decrease?

to make equally spaced vectors, use D = 0.25 / 2

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

P = 1.0 * np.pi

# define points on unit sphere which correspond to 4 clusters of each 2 NEARBY vectors
D = 0.25 / 2 * np.pi
c1 = np.array([

    P * 0.0 - D,
    P * 0.0 + D,

    P * 0.5 - D,
    P * 0.5 + D,

    P * 1.0 - D,
    P * 1.0 + D,

    P * 1.5 - D,
    P * 1.5 + D,
    ])

# define points on unit sphere which correspond to 4 clusters of each 2 OVERLAPPING vectors
D = 0.01 * np.pi
c2 = np.array([

    P * 0.0 - D,
    P * 0.0 + D,

    P * 0.5 - D,
    P * 0.5 + D,

    P * 1.0 - D,
    P * 1.0 + D,

    P * 1.5 - D,
    P * 1.5 + D,
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
    xy = np.hstack((x[:, np.newaxis], y[:, np.newaxis]))
    assert xy.shape == (len(c), 2)

    # calc cosine
    sim = cosine_similarity(xy)
    print(cosine_similarity(xy).mean())
    # print(cosine_similarity(xy).std())

    # calc average over off-diagonals (ignore 1s)
    masked = np.ma.masked_where(np.eye(*sim.shape), sim)
    print(masked.round(2))
    print(masked.mean())
    print()
