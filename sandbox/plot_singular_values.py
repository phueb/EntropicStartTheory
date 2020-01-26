import numpy as np

from startingabstract.figs import plot_singular_values

PARTITION_SIZE = 100  # size of imagined corpus partition
MAX_S = 30  # arbitrary value that is used to make singular value vectors the same length across simulations
PERFECT_STRUCTURE = True  # else use random variation to simulate an empirical corpus

d = np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
              [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]])

o = np.ones( (10, 10))
z = np.zeros((10, 10))

a = np.block(
    [
        [o, z],
        [z, o],
    ]
).astype(np.int)

b = np.block(
    [
        [o, z],
        [z, o],
    ]
).astype(np.int)

c = np.block(
    [
        [o, z],
        [z, o],
    ]
).astype(np.int)


def simulate_context_by_term_mat(mat):
    res = np.zeros_like(mat)
    nonzero_ids = np.nonzero(mat)
    num_nonzeros = np.sum(mat)
    for _ in range(PARTITION_SIZE):
        idx = np.random.choice(num_nonzeros, size=1)
        i = nonzero_ids[0][idx].item()
        j = nonzero_ids[1][idx].item()
        res[i, j] += 1

    # sums must match across matrices (because each partition has same num_tokens)
    assert res.sum() == PARTITION_SIZE

    return res


s_list = []
for legals_mat in [a[:-1], a[:-3], a[:-5], a[:-7], a[:-10]]:  # each mat represents a legals_mat

    print(legals_mat)

    # simulate co-occurrences from an imaginary corpus partition
    if PERFECT_STRUCTURE:
        ct_mat = legals_mat * (PARTITION_SIZE / np.sum(legals_mat))  # no randomness
    else:  # simulate random variation in co-occurrences
        ct_mat = simulate_context_by_term_mat(legals_mat)

    # SVD
    s = np.linalg.svd(ct_mat, compute_uv=False)
    print('svls', ' '.join(['{:>6.2f}'.format(si) for si in s]))
    print(np.sum(s))
    print()

    trailing = [np.nan] * (MAX_S - len(s))
    s_constant_length = np.hstack((s, trailing))
    s_list.append(s_constant_length)
    print(len(trailing))
    print(len(s_constant_length))

plot_singular_values(s_list, max_s=MAX_S)