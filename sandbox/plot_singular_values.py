import numpy as np

from startingabstract.figs import plot_singular_values

PARTITION_SIZE = 100  # size of imagined corpus partition
MAX_S = 20  # arbitrary value that is used to make singular value vectors the same length across simulations
PERFECT_STRUCTURE = True  # else use random variation to simulate an empirical corpus
SUBTRACT_MEAN = False  # if True, removes first singular dimension, coding for frequency
REMOVE_HALF_COLUMNS = False

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

o = np.ones( (6, 6))
z = np.zeros((6, 6))

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
for legals_mat in [a, np.ones_like(a)]:  # each mat represents a legals_mat

    if REMOVE_HALF_COLUMNS:
        legals_mat = legals_mat[:, :6]

    print(legals_mat)

    # simulate co-occurrences from an imaginary corpus partition
    if PERFECT_STRUCTURE:
        ct_mat = legals_mat * (PARTITION_SIZE / np.sum(legals_mat))  # no randomness
    else:  # simulate random variation in co-occurrences
        ct_mat = simulate_context_by_term_mat(legals_mat)

    if SUBTRACT_MEAN:
        ct_mat = ct_mat - ct_mat.mean().mean()

    # SVD
    u, s, v = np.linalg.svd(ct_mat, compute_uv=True)
    print('singular values:', ' '.join(['{:>6.2f}'.format(si) for si in s]))
    print('u')
    print(u.round(1))
    print('v')
    print(v.round(1))
    print()

    trailing = [np.nan] * (MAX_S - len(s))
    s_constant_length = np.hstack((s, trailing))
    s_list.append(s_constant_length)

plot_singular_values(s_list, max_s=MAX_S)