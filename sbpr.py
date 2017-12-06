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
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(x)
        return z / (1 + z)

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
            uranks[u, i] = rank(u, t, i, B, V_ui, V_iu, V_il, V_li)
    return uranks

def rank(u, t, i, B, V_ui, V_iu, V_il, V_li):
    """ Calculates item ranks for each user at time t.

    This calculates the rank of item i at time t
    for a given user u.

    Parameters
    ----------
    u : int
        Index of user u
    t : int
        Index of time t
    i : int
        Index of item i
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
    float
       The estimated rank
    """
    x = V_ui[u, :] @ V_iu[i, :].T
    NB = len(B[u][t-1])
    y = 0
    for l in B[u][t-1]:
        y =+ V_il[i, :] @ V_li[l, :].T
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
    np.array : np.float
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
    return np.array(index)


def update_user_matrix(u, t, i, j, B, V_ui, V_iu, V_li, V_il,
                       alpha, lam_ui, lam_iu):
    """ Updates the user parameters

    Parameters
    ----------
    u : int
        Index for user.
    t : int
        Time point.
    i : int
        Index for item i for preferred items.
    j : int
        Index for item j for probably not preferred items.
    B : list of list of list
        Basket of items for each user u at a specific time t
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
    V_ui : np.array
        Factor matrix for users u to items i (if inplace is False)
    V_iu : np.array
        Factor matrix for items i to users u (if inplace is False)

    Note
    ----
    There are side effects.  So V_ui and V_iu are modified in place
    and not actually returned.  This is done for the sake of optimization.
    """
    ri = rank(u, t, i, B, V_ui, V_iu, V_li, V_il)
    rj = rank(u, t, j, B, V_ui, V_iu, V_li, V_il)
    delta = 1 - sigmoid(ri - rj)

    for f in range(V_iu.shape[1]):
        V_ui[u, f] += alpha * (
            delta * (V_iu[i, f] - V_iu[j, f]) - lam_ui * V_ui[u, f]
        )
        V_iu[i, f] += alpha * (
            delta * V_ui[u, f] - lam_iu * V_iu[i, f]
        )
        V_iu[j, f] += alpha * (
            -delta * V_ui[u, f] - lam_iu * V_iu[j, f]
        )


def update_item_matrix(u, t, i, j, B, V_ui, V_iu, V_li, V_il,
                       alpha, lam_il, lam_li):
    """ Updates the item parameters

    Parameters
    ----------
    u : int
        Index for user.
    t : int
        Time point.
    i : int
        Index for item i for preferred items.
    j : int
        Index for item j for probably not preferred items.
    B : list of list of list
        Triplely nested lists representing sets index by user and time
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
    V_li : np.array
        Factor matrix for items l to items i
    V_il : np.array
        Factor matrix for items i to items l

    Note
    ----
    There are side effects.  So V_li and V_il are modified in place
    and not actually returned.  This is done for the sake of optimization.
    """
    ri = rank(u, t, i, B, V_ui, V_iu, V_li, V_il)
    rj = rank(u, t, j, B, V_ui, V_iu, V_li, V_il)
    delta = 1 - sigmoid(ri - rj)

    for f in range(V_il.shape[1]):
        NB = len(B[u][t]) # get size of the basket
        eta = np.sum([V_li[l, f] for l in B[u][t]]) / NB

        V_il[i, f] += alpha * (delta * eta - lam_il * V_il[i, f])
        V_il[j, f] += alpha * (-delta * eta - lam_il * V_il[j, f])
        for l in B[u][t]:
            V_li[l, f] += alpha * (delta * (V_il[i, f] - V_il[j, f]) / NB -
                                   lam_li * V_li[l, f])

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
    NB = [2**((-i - 1) / (alpha - 1)) for i in range(len(B))]
    score = 100 * score / NB

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
    """
    top = np.argsort(r)[:N]
    hits = set([I[i] for i in top]) & B
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
    return score

def score(rs, Bs, I, method='auc'):
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
    s = []
    for u in range(len(Bs)):
        # get last timepoint
        s.append(auc(rs[u], Bs[u][-1], I))
    return np.mean(s)


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
        num_reviews = poisson(lamR)
        # starting position
        s = randint(0, kI)
        B[u] = [[] for _ in range(num_reviews)]
        for t in range(num_reviews):
            # random walk to next basket
            num_items = poisson(lamI)
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


