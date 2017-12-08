import pandas as pd
import numpy as np
from numba import jit
from scipy.sparse import coo_matrix, csr_matrix
from numpy.random import poisson, randint, random


def LearnSBPR_FPMC(S):
    """ Perform sequential personalized ranking.

    Parameters
    ----------
    S : scipy sparse matrix
        Sparse matrix of presence only information.
        This typically represents user/item interactions

    Returns
    -------
    V_ui : np.array
        Factor matrix for users u to items i
    V_iu : np.array
        Factor matrix for items i to users u
    V_il : np.array
        Factor matrix for items i to items l
    V_li : np.array
        Factor matrix for items l to items i
    """
    pass


@jit(nopython=True, cache=True)
def sigmoid(x):
    "Numerically-stable sigmoid function."
    if x >= 0:
        z = np.exp(-x)
        res = 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        res = z / (1 + z)

    if res > 1e-100:
        return res
    else:
        return 1e-100

@jit(nopython=True, cache=True)
def rank(boot, e, V_ui, V_iu, V_il, V_li):
    """ Calculates item ranks for each user at time t.

    This calculates the rank of item i at time t
    for a given user u.

    Parameters
    ----------
    boot : np.array
        Contains (u, t, i, j, l) tuples
        One instance of (u, t, i, j).  Multiple instances
        of l.  u = user, t = time point, i = item, j=unrated item,
        l = other items in basket.
    e : bool
        Indicotor if item i should be ranked, or item j
    V_ui : np.array
        Left factor matrix for users u to items i
    V_iu : np.array
        Right factor matrix for items i to users u
    V_il : np.array
        Left factor matrix for items i to items l
    V_li : np.array
        Right Factor matrix for items l to items i

    Returns
    -------
    float
       The estimated rank
    """
    u, t, i, j, l = boot[0, :]
    # stupid hack to get ranks working
    if e:
        p = j
    else:
        p = i
    x = V_ui[u, :] @ V_iu[p, :].T
    NB = len(boot)
    y = 0
    for k in range(NB):
        u, t, _, _, l = boot[k]
        y += V_il[p, :] @ V_li[l, :].T
    return x + y / NB


def bootstrap(B, I, n):
    """ Bootstraps (u, t, i, j) tuples.

    Parameters
    ----------
    B : list of list of list
        users by time by basket
        items for a specific user at a given time.
    I : set
        Total set of items
    n : int
        Number of bootstraps

    Returns
    -------
    list of tuple
        List of (u, t, i, j) tuples.
    """
    # construct indexes
    index = []
    for _ in range(n):
        # random user
        u = np.random.randint(0, len(B))
        # random time (except first timepoint)
        t = np.random.randint(1, len(B[u]))
        # random rated item
        i_ = np.random.randint(0, len(B[u][t]))
        i = B[u][t][i_]
        # random unrated item
        j_ = np.random.randint(0, len(I))
        j = I[j_]
        while j in set(B[u][t]):
            j_ = np.random.randint(0, len(I))
            j = I[j_]
        index.append((u, t, i, j))
    return index

@jit(nopython=True)
def fast_bootstrap(B, I, n):
    """ Bootstraps (u, t, i, j, l) tuples.

    Parameters
    ----------
    B : np.array
        DataFrame where column are
        [user_id, order_id, order_number, product_id]
    I : np.array
        Total set of items
    n : int
        Number of bootstraps
    seed : int
        Random seed

    Returns
    -------
    list of tuple
        List of (u, t, i, j, l) tuples.

    Note
    ----
    There is something weird about the random seed ...
    So every result is random from here on out :(
    """
    # get top n random indexes
    idx = np.random.choice(np.arange(len(B)), replace=False, size=n)

    # get j for each entry
    for oi in idx:

        u = B[oi, 0]  # user id
        o = B[oi, 1]  # order id
        t = B[oi, 2]  # time
        i = B[oi, 3]  # time

        # find bounds of basket o
        lo, hi = 0, 0
        while (oi - lo) > 0:
            u_ = B[oi - lo - 1, 0]  # user id
            o_ = B[oi - lo - 1, 1]  # order id
            t_ = B[oi - lo - 1, 2]  # time
            if u_ != u or o_ !=o or t_ != t: break
            lo += 1

        while (oi + hi) < len(B) - 1:
            u_ = B[oi + hi + 1, 0]  # user id
            o_ = B[oi + hi + 1, 1]  # order id
            t_ = B[oi + hi + 1, 2]  # time
            if u_ != u or o_ !=o or t_ != t: break
            hi += 1

        items = [B[i, 3] for i in range(oi - lo, oi + hi + 1)]

        # get unpreferred item
        j_ = np.random.randint(0, len(I))
        j = I[j_]

        while j in items:
            j_ = np.random.randint(0, len(I))
            j = I[j_]

        res = np.zeros((len(items), 5), dtype=np.int64)
        # get l for each entry
        for k, l in enumerate(items):
            res[k] = np.array([u, t, i, j, l], dtype=np.int64)
        yield res


@jit(nopython=True, cache=True)
def update_user_matrix(boot, dV_ui, dV_iu,
                       V_ui, V_iu, V_li, V_il,
                       alpha, lam_ui, lam_iu):
    """ Updates the user parameters

    Parameters
    ----------
    boot : np.array
        Contains (u, t, i, j, l) tuples
        One instance of (u, t, i, j).  Multiple instances
        of l.  u = user, t = time point, i = item, j=unrated item,
        l = other items in basket.
    dV_ui : np.array
        Factor update matrix for users u to items i
    dV_iu : np.array
        Factor update matrix for items i to users u
    V_ui : np.array
        Factor matrix for users u to items i
    V_iu : np.array
        Factor matrix for items i to users u
    V_il : np.array
        Factor matrix for users u to items i
    V_li : np.array
        Factor matrix for items i to users u
    alpha : np.float
        Learning rate
    lam_ui : np.array
        Regularization constant for V_ui factor
    lam_iu : np.array
        Regularization constant for V_iu factor

    Returns
    -------
    dV_ui : np.array
        Factor update matrix for users u to items i (if inplace is False)
    dV_iu : np.array
        Factor update matrix for items i to users u (if inplace is False)

    Note
    ----
    There are side effects.  So V_ui and V_iu are modified in place
    and not actually returned.  This is done for the sake of optimization.
    """
    u, t, i, j, l = boot[0]
    ri = rank(boot, False, V_ui, V_iu, V_li, V_il)
    rj = rank(boot, True, V_ui, V_iu, V_li, V_il)

    delta = 1 - sigmoid(ri - rj)
    for f in range(V_iu.shape[1]):
        dV_ui[u, f] += alpha * (
            delta * (V_iu[i, f] - V_iu[j, f]) - lam_ui * V_ui[u, f]
        )
        dV_iu[i, f] += alpha * (
            delta * V_ui[u, f] - lam_iu * V_iu[i, f]
        )
        dV_iu[j, f] += alpha * (
            -delta * V_ui[u, f] - lam_iu * V_iu[j, f]
        )

@jit(nopython=True, cache=True)
def update_item_matrix(boot, dV_li, dV_il,
                       V_ui, V_iu, V_li, V_il,
                       alpha, lam_il, lam_li):
    """ Updates the item parameters

    Parameters
    ----------
    boot : np.array
        Contains (u, t, i, j, l) tuples
        One instance of (u, t, i, j).  Multiple instances
        of l.  u = user, t = time point, i = item, j=unrated item,
        l = other items in basket.
    dV_il : np.array
        Factor update matrix for users u to items i
    dV_li : np.array
        Factor update matrix for items i to users u
    V_ui : np.array
        Factor matrix for users u to items i
    V_iu : np.array
        Factor matrix for items i to users u
    V_il : np.array
        Factor matrix for users u to items i
    V_li : np.array
        Factor matrix for items i to users u
    alpha : np.float
        Learning rate
    lam_il : np.array
        Regularization constant for V_il factor
    lam_li : np.array
        Regularization constant for V_li factor

    Returns
    -------
    dV_li : np.array
        Factor update matrix for items l to items i
    dV_il : np.array
        Factor update matrix for items i to items l

    Note
    ----
    There are side effects.  So V_li and V_il are modified in place
    and not actually returned.  This is done for the sake of optimization.
    """
    u, t, i, j, l = boot[0]
    ri = rank(boot, False, V_ui, V_iu, V_li, V_il)
    rj = rank(boot, True, V_ui, V_iu, V_li, V_il)

    delta = 1 - sigmoid(ri - rj)

    for f in range(V_il.shape[1]):
        NB = len(boot) # get size of the basket
        eta = 0
        for k in range(len(boot)):
            u, t, i, j, l = boot[k]
            eta += V_li[l, f]
        eta = eta / NB

        dV_il[i, f] += alpha * (delta * eta - lam_il * V_il[i, f])
        dV_il[j, f] += alpha * (-delta * eta - lam_il * V_il[j, f])
        for k in range(len(boot)):
            u, t, i, j, l = boot[k]
            dV_li[l, f] += alpha * ((delta * (V_il[i, f] - V_il[j, f]) / NB) -
                                   lam_li * V_li[l, f])


def cost(boot, V_ui, V_iu, V_li, V_il,
         lam_ui, lam_iu, lam_il, lam_li):
    """
    Calculates log (sigmoid(z_uti - z_utj))

    Parameters
    ----------
    boot : np.array
        Contains (u, t, i, j, l) tuples
        One instance of (u, t, i, j).  Multiple instances
        of l.  u = user, t = time point, i = item, j=unrated item,
        l = other items in basket.
    V_ui : np.array
        Factor matrix for users u to items i
    V_iu : np.array
        Factor matrix for items i to users u
    V_il : np.array
        Factor matrix for items i to items l
    V_li : np.array
        Factor matrix for items l to items i
    lam_iu : np.array
        Regularization constant for V_iu factor
    lam_ui : np.array
        Regularization constant for V_ui factor
    lam_il : np.array
        Regularization constant for V_il factor
    lam_li : np.array
        Regularization constant for V_li factor

    Return
    ------
    np.float
        Cost for that particular bootstrap.
    """
    ri = rank(boot, False, V_ui, V_iu, V_li, V_il)
    rj = rank(boot, True, V_ui, V_iu, V_li, V_il)
    delta = np.log(sigmoid(ri - rj))
    return delta


# @jit(nopython=True, cache=True)
def user_item_ranks(B, V_ui, V_iu, V_il, V_li):
    """ Calculates item ranks for each user at last time point t.

    This calculates the rank of item i at time t
    for a given user u.

    Parameters
    ----------
    B : np.array
        DataFrame where column are
        [user_id, order_id, order_number, product_id]
        It is assumed that only one timepoint is considered.
        It is also assumed that all of the entries are ordered by
        user_id and order_number.
    V_ui : np.array
        Left factor matrix for users u to items i
    V_iu : np.array
        Right factor matrix for items i to users u
    V_il : np.array
        Left factor matrix for items i to items l
    V_li : np.array
        Right Factor matrix for items l to items i

    Returns
    -------
    np.array
       User by item matrix
    """
    kU = V_ui.shape[0]
    kI = V_iu.shape[0]
    uranks = np.zeros((kU, kI))

    res = []
    oi = 0
    u = B[oi, 0]  # user id
    o = B[oi, 1]  # order id
    t = B[oi, 2]  # time
    i = B[oi, 3]  # item
    o_ = B[oi, 1] # order id
    res.append((u, t, i, 0, i))
    for oi in range(B.shape[0]):
        if o != o_:
            o_ = B[oi, 1]
            r = rank(np.array(res), False, V_ui, V_iu, V_il, V_li)
            uranks[u, i] = r
            u = B[oi, 0]  # user id
            o = B[oi, 1]  # order id
            t = B[oi, 2]  # time
            i = B[oi, 3]  # item
            l = B[oi, 3]  # item
            res = []
            res.append((u, t, i, 0, l))
        else:
            l = B[oi, 3]  # item
            o = B[oi, 1]  # order id
            res.append((u, t, i, 0, l))
    return uranks

@jit(nopython=True, cache=True)
def _auc(r, B, I):
    """ Calculates AUC for a single user.

    Parameters
    ----------
    r : np.array
        Ranks for items for a user u at time t
    B : list
        A basket for user u
    I : set
        Set of items

    Parameters
    ----------
    score : float
        Fraction of correct ranks
    """
    IB = I - set(B)
    NIB = len(IB)
    NB = len(B)
    score = 0
    for i in B:
        for j in IB:
            score += int(r[i] < r[j])
    return score / (NB * NIB)

def auc(R, B, I):
    """ Calculates scores for all users

    Parameters
    ----------
    R : np.array
        User ranks for kU users and kI items
    B : np.array
        DataFrame where column are
        [user_id, order_id, order_number, product_id]
        This is assuming to only operate on one timepoint.
    I : list
        List of items

    Returns
    ----------
    s : list, float
        List of AUC scores
    """
    res = []
    oi = 0
    u = B[oi, 0]   # user id
    o = B[oi, 1]   # order id
    t = B[oi, 2]   # time
    i = B[oi, 3]   # item
    o_ = B[oi, 1]  # order id
    I_ = set(I)
    s = []
    k = 0
    res.append((u, t, i, 0, i))
    for oi in range(B.shape[0]):
        if o != o_:
            o_ = B[oi, 1]
            b = np.array(res, dtype=np.int64)
            s.append( _auc(R[u, :], b[:, 3], I_) )

            u = B[oi, 0]  # user id
            o = B[oi, 1]  # order id
            t = B[oi, 2]  # time
            i = B[oi, 3]  # item
            l = B[oi, 3]  # item
            res = []
            res.append((u, t, i, 0, l))
            k += 1
        else:
            l = B[oi, 3]  # item
            o = B[oi, 1]  # order id
            res.append((u, t, i, 0, l))
    return s

# checks
def hlu(r, B, I, alpha):
    """
    Parameters
    ----------
    r : np.array
        Ranks for items for a user u at time t
    B : list
        A basket for user u
    I : list
        List of items
    alpha : float
        Half-life parameter.

    Returns
    -------
    float :
        Breese score
    """
    score = 0
    for i in I:
        if r[i] > 0:
            score += 2**((-i - 1) / (alpha - 1))
    NB = np.sum([2**((-i - 1) / (alpha - 1)) for i in range(len(B))])
    score = 100 * score / NB
    return score

def top_precision_recall(r, B, I, N):
    """
    Parameters
    ----------
    r : np.array
        Ranks for items for a user u at time t
    B : list
        A basket for user u
    I : list
        List of items
    N : int
        Top N ranks

    Returns
    -------
    precision : float
        Precision score
    recall : float
        Recall score

    Note
    ----
    This is broken!!!
    """
    top = np.argsort(r)[:N]
    hits = set([I[i] for i in top]) & set(B)
    prec = len(hits) / N
    recall = len(hits) / len(B)
    return prec, recall

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# utilities
def simulate(kU=10, kI=11, mu=0, scale=1, lamR=15, lamI=5):
    """

    Parameters
    ----------
    kU : int
       Number of users
    kI : int
       Number of items
    mu : float
       Mean of transition probabilities
    scale : float
       Scaling of transition probabilities
    lamR : int
       Average number of reviews per user
    lamI : int
       Average number of items per basket

    Returns
    -------
    trans_probs : np.array
       Transition probabilities between any two items.
    orders : pd.DataFrame
       DataFrame of orders and users who have made the order,
       along with the product ids. The column names are
       [user_id, order_id, order_number, product_id]
    """
    kL = kI

    trans_probs = np.zeros((kU, kL, kI))
    idx_prob = np.zeros((kU, kL, kI))

    for u in range(kU):
        for i in range(kL):
            trans_probs[u, i, :] = softmax(np.random.normal(size=kI))
            idx_prob[u, i, :] = np.cumsum(trans_probs[u, i, :])

    B = []
    for u in range(kU):
        num_reviews = poisson(lamR) + 1
        # starting position
        s = randint(0, kI)

        for t in range(num_reviews):
            # random walk to next basket
            num_items = poisson(lamI) + 1
            for i in range(num_items):
                R = random()
                # get next state
                s = np.searchsorted(idx_prob[u, s, :], R)
                B.append((u, int('1%s%s' % (u, t)), t, s))
    orders = pd.DataFrame(B, columns=['user_id', 'order_id',
                                      'order_number', 'product_id'])
    return trans_probs, orders


