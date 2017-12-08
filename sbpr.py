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

def user_item_ranks(B, V_ui, V_iu, V_il, V_li):
    """ Calculates item ranks for each user at last time point t.

    This calculates the rank of item i at time t
    for a given user u.

    Parameters
    ----------
    B : list of list of list
        Basket of items for each user u at a specific time t
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
    for u in range(len(B)):
        t = len(B[u])
        for i in B[u][t-1]:
            res = []
            for l in B[u][t-1]:
                res.append((u, t, i, 0, l))
            res = np.array(res)
            uranks[u, i] = rank(res, False, V_ui, V_iu, V_il, V_li)
    return uranks

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

def fast_bootstrap(B, P, I, n):
    """ Bootstraps (u, t, i, j, l) tuples.

    Parameters
    ----------
    B : pd.DataFrame
        DataFrame where column are
        [order_id, order_number, user_id]
        It is assumed that the order_id is the index
    P : pd.DataFrame
        DataFrame where columns are
        [order_id, product_id]
        It is assumed that the order_id is the index
    I : set
        Total set of items
    n : int
        Number of bootstraps

    Returns
    -------
    list of tuple
        List of (u, t, i, j, l) tuples.
    """
    # get top n random indexes
    idx = np.random.choice(B.index, replace=False, size=n)
    subB = B.loc[idx]
    # get i for each entry
    for oi in subB.index:

        u = subB.loc[oi][0]  # user
        t = subB.loc[oi][1]  # time

        items = P.loc[oi].values
        i_ = np.random.randint(0, len(items))
        i = items[i_]
        j_ = np.random.randint(0, len(I))
        j = I[j_]

        while j in set(items.ravel().tolist()):
            j_ = np.random.randint(0, len(I))
            j = I[j_]
        res = []
        # get l for each entry
        for l in items:
            res.append((u, t, i, j, l))
        yield np.array(res)

def flatten(index, B):
    """
    Parameters
    ----------
    index : np.array
       List of (u, t, i, j) tuples.
    B : list of list of list
        Basket of items for each user u at a specific time t

    Returns
    -------
    iterable of np.array
       List of (u, t, i, j, l, it) tuples.
    """

    for it in range(len(index)):
        res = []
        u, t, i, j = index[it]
        for l in B[u][t]:
            res.append((u, t, i, j, l))
        yield np.array(res)

def update(X, i, j, dX, clip):
    """ Update parameter

    Parameters
    ----------
    X : np.array
       Parameter to be updated
    i : int
       Row index
    j : int
       Column index
    dX : np.array
       Change in parameter
    clip : float
       Clipping threshold to prevent exploding gradients.
    """

    X[i, j] = dX

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

def auc(r, B, I):
    """ Calculates AUC for a single user.

    Parameters
    ----------
    r : np.array
        Ranks for items for a user u at time t
    B : list
        A basket for user u
    I : list
        List of items

    Parameters
    ----------
    score : float
        Fraction of correct ranks
    """
    IB = set(I) - set(B)
    NIB = len(IB)
    NB = len(B)
    score = 0
    for i in B:
        for j in IB:
            score += int(r[i] < r[j])
    return score / (NB * NIB)

def score(rs, Bs, I, alpha=0.5, N=10, method='auc'):
    """ Calculates scores for all users

    Parameters
    ----------
    r : np.array
        Ranks for items for a user u at time t
    Bs : list
        List of baskets for each user u
    I : list
        List of items
    method : str
        Scoring method.

    Parameters
    ----------
    s : float
        Fraction of correct ranks
    """
    if method == 'auc':
        s = []
        for u in range(len(Bs)):
            # get last timepoint
            s.append(auc(rs[u], Bs[u][-1], I))
        return np.mean(s)
    if method == 'hlu':
        s = [0] * len(Bs)
        for u in range(len(Bs)):
            # get last timepoint
            s[u] = hlu(rs[u], Bs[u][-1], I, alpha)
        return np.mean(s)
    if method == 'top_precision_recall':
        prec, recall = [], []
        for u in range(len(Bs)):
            # get last timepoint
            p, r = top_precision_recall(rs[u], Bs[u][-1], I, N)
            prec.append(p)
            recall.append(r)
        return np.mean(prec), np.mean(recall)


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
    """
    kL = kI

    trans_probs = np.zeros((kU, kL, kI))
    idx_prob = np.zeros((kU, kL, kI))

    for u in range(kU):
        for i in range(kL):
            trans_probs[u, i, :] = softmax(np.random.normal(size=kI))
            idx_prob[u, i, :] = np.cumsum(trans_probs[u, i, :])

    B = [[] for _ in range(kU)]
    for u in range(kU):
        num_reviews = poisson(lamR) + 1
        # starting position
        s = randint(0, kI)
        B[u] = [[] for _ in range(num_reviews)]
        for t in range(num_reviews):
            # random walk to next basket
            num_items = poisson(lamI) + 1
            for i in range(num_items):
                R = random()
                # get next state
                s = np.searchsorted(idx_prob[u, s, :], R)
                B[u][t] += [s]
    return trans_probs, B


def split_dataset(B):
    """ Creates a training and test dataset.

    The training dataset contains all baskets
    up to the last time point for each user.
    The test dataset contains the last basket for
    each user.

    Parameters
    ----------
    B : list of list of list
        users by time by basket
        items for a specific user at a given time.

    Returns
    -------
    train : list of list of list
        Training baskets
    test : list of list
        Test baskets
    """
    train = [[] for _ in range(len(B))]
    for u in range(len(B)):
        train[u] = B[u][:-1]

    test = [[] for _ in range(len(B))]
    for u in range(len(B)):
        test[u] = [B[u][-1]]
    return train, test


