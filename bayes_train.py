import numpy as np
import numpy.testing as npt
from numpy.random import poisson, randint, random
from numpy.linalg import norm
from scipy.sparse import csr_matrix
from sbpr import (fast_bootstrap, update_user_matrix, update_item_matrix,
                  simulate, user_item_ranks,
                  cost, rank, sigmoid, f1_score, predict)
import copy
from bayes_opt import BayesianOptimization

from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import pickle

prior_products = pd.read_csv('order_products__prior.csv')
train_products = pd.read_csv('order_products__train.csv')
orders = pd.read_csv('orders.csv')
prior_orders = orders.loc[orders.eval_set=='prior']
train_orders = orders.loc[orders.eval_set=='train']
user_orders = prior_orders.user_id.value_counts()
product_orders = prior_products.product_id.value_counts()

## removing users
user_set = set(user_orders.loc[user_orders > 50].index)
prior_orders = prior_orders.loc[
    [i in user_set for i in prior_orders.user_id]
]
train_orders = train_orders.loc[
    [i in user_set for i in train_orders.user_id]
]

## removing items
product_set = set(product_orders.loc[product_orders > 1000].index)
prior_products = prior_products.loc[
    [i in product_set for i in prior_products.product_id]
]
train_products = train_products.loc[
    [i in product_set for i in train_products.product_id]
]

A = prior_orders[['user_id', 'order_id', 'order_number']].sort_values('order_id')
B = prior_products[['order_id', 'product_id']].sort_values('order_id')
C = pd.merge(A, B, on='order_id')
prior = C.sort_values(by=['user_id', 'order_number']) - 1 # account for 1 offset

A = train_orders[['user_id', 'order_id', 'order_number']].sort_values('order_id')
B = train_products[['order_id', 'product_id']].sort_values('order_id')
C = pd.merge(A, B, on='order_id')
train = C.sort_values(by=['user_id', 'order_number']) - 1 # account for 1 offset


le_user = preprocessing.LabelEncoder()
le_product = preprocessing.LabelEncoder()

le_product.fit(prior.product_id)
le_user.fit(prior.user_id)

prior['user_id'] = le_user.transform(prior['user_id'])
train['user_id'] = le_user.transform(train['user_id'])

prior['product_id'] = le_product.transform(prior['product_id'])
train['product_id'] = le_product.transform(train['product_id'])


from numba import jit
@jit(nopython=True, cache=True)
def stupid_query(X, sizes):
    idx = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        u = X[i, 0]
        o = X[i, 2]
        idx[i] = (o == (sizes[u] - 1))
    return idx

sizes = prior.groupby(['user_id', 'order_id']).size()
user_orders = [len(sizes[i]) for i in range(len(np.unique(prior.user_id)))]
idx = stupid_query(prior.values, np.array(user_orders))
idx = idx.astype(np.bool)

prior_train = prior.iloc[~idx]
prior_test = prior.iloc[idx]

num_boots = 10000
num_iters = 100


def target(rUI=10, rIL=10, lam_ui=1, lam_iu=1, lam_il=1, lam_li=1,
           clip=2, alpha=1e-1, decay=0.9):
    kU = len(prior.user_id.unique())
    kI = len(prior.product_id.unique())
    kL = kI

    #lam_ui=9
    #lam_iu=7
    #lam_il=5
    #lam_li=0.5
    #clip = 10
    #alpha = 1e-4
    #decay=0.95
    verbose = True

    rUI = int(rUI)
    rIL = int(rIL)
    V_ui = np.random.normal(size=(kU, rUI))
    V_iu = np.random.normal(size=(kI, rUI))
    V_li = np.random.normal(size=(kI, rIL))
    V_il = np.random.normal(size=(kI, rIL))

    I = list(range(kI))

    for b in range(num_iters):
        gen = list(fast_bootstrap(prior_train.values, I, num_boots))

        c = 0 # cost
        dV_ui = np.zeros(shape=(kU, rUI))
        dV_iu = np.zeros(shape=(kI, rUI))
        dV_li = np.zeros(shape=(kI, rIL))
        dV_il = np.zeros(shape=(kI, rIL))

        for boot in gen:
            update_user_matrix(boot, dV_ui, dV_iu,
                               V_ui, V_iu, V_li, V_il,
                               alpha=alpha, lam_ui=lam_ui, lam_iu=lam_iu)

            update_item_matrix(boot, dV_li, dV_il,
                               V_ui, V_iu, V_li, V_il,
                               alpha=alpha, lam_il=lam_il, lam_li=lam_li)

        # gradient clipping:
        # http://www.wildml.com/deep-learning-glossary/#gradient-clipping
        V_ui += (dV_ui * clip) / norm(dV_ui)
        V_iu += (dV_iu * clip) / norm(dV_iu)
        V_li += (dV_li * clip) / norm(dV_li)
        V_il += (dV_il * clip) / norm(dV_il)
        cs = []
        for boot in gen:
            c = cost(boot,
                      V_ui, V_iu, V_li, V_il,
                      lam_ui=0, lam_iu=0,
                      lam_il=0, lam_li=0)
            cs.append(c)

        if np.mean(cs) > -0.01: break
        alpha *= decay
        if b % 200 == 0 and verbose:
            print('cost %3.3f' % np.mean(cs),
                  'V_ui [%3.3f, %3.3f]'   % (V_ui.min(), V_ui.max()),
                  'V_iu [%3.3f, %3.3f]'   % (V_iu.min(), V_iu.max()),
                  'V_li [%3.3f, %3.3f]'   % (V_li.min(), V_li.max()),
                  'V_il [%3.3f, %3.3f]'   % (V_il.min(), V_il.max()))
    return np.mean(cs)

bo = BayesianOptimization(target,
                          {'rUI':(1, 100),
                           'rIL':(1, 100),
                           'lam_ui': (0, 100),
                           'lam_iu': (0, 100),
                           'lam_il': (0, 100),
                           'lam_li': (0, 100),
                           'clip': (1, 10),
                           'alpha': (1e-6, 1),
                           'decay': (0.7, 1)})
bo.maximize(init_points=5, n_iter=5, acq='ucb', kappa=4)

print(bo.res['max'])
print(bo.res['all'])
