import numpy as np

ONE = np.ones((1, 1), dtype=np.bool_)
ZERO = np.zeros((1, 1), dtype=np.bool_)


def choose(n: int, k: int):
    """Generate all k-choices of n elements as a binary array."""
    assert 0 <= k <= n

    if k == 0:
        yield np.zeros(n, dtype=np.bool_)
    elif k == n:
        yield np.ones(n, dtype=np.bool_)
    else:
        for choice in choose(n - 1, k - 1):
            yield np.append(choice, 1)
        for choice in choose(n - 1, k):
            yield np.append(choice, 0)


def structured_mask(n: int, m: int, nk: np.ndarray, mk: int):
    """Generate a structured binary mask.

    The mask is a n by m binary matrix. A structured mask is a mask where
    row i contains exactly nk[i] ones and column j contains exactly mk ones.

    We recursively yield all such possible masks.
    """
    if n < mk:  # Not enough rows to fill in ones.
        return

    if np.any(m < nk):  # Not enough columns to fill in ones.
        return

    if n == mk and np.all(nk == m):
        yield np.ones((n, m), dtype=np.bool_)
        return

    empty_rows = nk == 0
    if np.any(empty_rows):
        # Call recursively on the remaining rows and fill in the empty rows.
        n_empty = np.sum(empty_rows)
        for mask in structured_mask(n - n_empty, m, nk[~empty_rows], mk):
            out = np.zeros((n, m), dtype=np.bool_)
            out[~empty_rows] = mask
            yield out
    else:
        # Each row has at least one element.
        for choice in choose(n, mk):
            # print(f"{choice=} {n=} {m=} {nk=} {mk=}")
            for mask in structured_mask(n, m - 1, nk - choice, mk):
                out = np.zeros((n, m), dtype=np.bool_)
                out[:, -1] = choice
                out[:, :-1] = mask
                # print(f"{out=}")
                yield out


def balanced_mask(n: int, m: int, nk: int):
    assert n * nk % m == 0
    mk = n * nk // m
    yield from structured_mask(n, m, np.full(n, nk), mk)


if __name__ == "__main__":
    # for a in choose(4, 2):
    #     print(a)

    count = 0
    for a in balanced_mask(8, 4, 2):
        # print(a.astype(np.int32))
        count += 1

    print(count)
