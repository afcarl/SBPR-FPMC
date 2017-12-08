import unittest
import numpy as np
import pandas as pd
import numpy.testing as npt
from numpy.random import poisson, randint, random
from scipy.sparse import csr_matrix
from sbpr import (bootstrap, update_user_matrix, update_item_matrix,
                  simulate, user_item_ranks, auc,
                  fast_bootstrap, cost)
import copy


# will use the instacart dataset
# https://www.kaggle.com/c/instacart-market-basket-analysis
class TestSBPR(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        # generate a quasi-realistic generative model
        # this will involve randomly generating the transition tensor
        kU = 10
        kI = 11
        kL = 11
        trans_probs, df = simulate(kU=kU, kI=kI,
                                  mu=0, scale=1,
                                  lamR=15, lamI=5)
        self.B = df
        self.trans_probs = trans_probs
        self.kU = kU   # number of users
        self.kL = kL   # number of items
        self.kI = kI   # number of items

        # setting training vs testing data
        order_lookup = df.order_id.value_counts()
        sizes = df.groupby(['user_id', 'order_id']).size()
        def is_train(x):
            if x['order_number'] == len(sizes.loc[x['user_id']]) - 1:
                return 'test'
            else:
                return 'train'
        df['eval_type'] = df.apply(is_train, axis=1)

        self.train = df.loc[df.eval_type=='train',
                            ['user_id', 'order_id',
                             'order_number', 'product_id']]
        self.test = df.loc[df.eval_type=='test',
                           ['user_id', 'order_id',
                            'order_number', 'product_id']]


    def test_simulation(self):
        np.random.seed(0)
        trans_probs, B = simulate(kU=3, kI=3,
                                  mu=0, scale=1,
                                  lamR=2, lamI=2)
        exp = np.array(
            [[0, 100, 0, 2],
             [0, 100, 0, 1],
             [0, 101, 1, 1],
             [0, 101, 1, 1],
             [0, 102, 2, 1],
             [0, 102, 2, 1],
             [0, 103, 3, 0],
             [0, 103, 3, 1],
             [0, 103, 3, 0],
             [0, 103, 3, 1],
             [0, 103, 3, 0],
             [0, 104, 4, 1],
             [0, 104, 4, 1],
             [0, 104, 4, 1],
             [0, 104, 4, 1],
             [1, 110, 0, 0],
             [1, 110, 0, 0],
             [1, 111, 1, 1],
             [1, 111, 1, 0],
             [2, 120, 0, 0],
             [2, 120, 0, 0],
             [2, 120, 0, 1],
             [2, 120, 0, 1],
             [2, 120, 0, 1],
             [2, 120, 0, 0]])
        npt.assert_allclose(B, exp)

        exp = np.array([[[0.58423523, 0.14936733, 0.26639744],
                         [0.57854881, 0.39829292, 0.02315827],
                         [0.59482294, 0.19771333, 0.20746373]],
                        [[0.21712474, 0.16632057, 0.61655469],
                         [0.44329537, 0.2338953, 0.32280932],
                         [0.20943928, 0.66836778, 0.12219294]],
                        [[0.73090254, 0.22749237, 0.04160509],
                         [0.40284797, 0.4973912, 0.09976083],
                         [0.88315015, 0.02131423, 0.09553562]]])

        npt.assert_allclose(trans_probs, exp, rtol=1e-4, atol=1e-4)

    def test_fast_bootstrap(self):
        np.random.seed(0)
        X = pd.DataFrame(
            [
                [202279, 2, 3, 33120],
                [202279, 2, 3, 28985],
                [202279, 2, 3, 9327],
                [202279, 2, 3, 45918],
                [202279, 2, 3, 30035],
                [202279, 2, 3, 17794],
                [202279, 2, 3, 40141],
                [202279, 2, 3, 1819],
                [202279, 2, 3, 43668],
                [205970, 3, 16, 32665],
                [205970, 3, 16, 17461],
                [205970, 3, 16, 46667],
                [205970, 3, 16, 17668],
                [205970, 3, 16, 17704],
                [205970, 3, 16, 24838],
                [205970, 3, 16, 33754],
                [205970, 3, 16, 21903],
                [178520, 4, 36, 40285],
                [178520, 4, 36, 41276],
                [178520, 4, 36, 32645]],
            columns=['user_id', 'order_id',
                     'order_number', 'product_id'],
        )
        I = X.product_id.values
        gen = fast_bootstrap(X.values, I, 20)
        list(gen)  # make sure that there are no out of bounds errors.

    def test_fast_bootstrap2(self):
        np.random.seed(0)
        X = pd.DataFrame(
            [
                [202279, 2, 3, 33120],
                [202279, 2, 3, 28985],
                [202279, 2, 3, 9327],
                [202279, 2, 3, 45918],
                [202279, 2, 3, 30035],
                [202279, 2, 3, 17794],
                [202279, 2, 3, 40141],
                [202279, 2, 3, 1819],
                [202279, 2, 3, 43668],
                [205970, 3, 16, 32665],
                [205970, 3, 16, 17461],
                [205970, 3, 16, 46667],
                [205970, 3, 16, 17668],
                [205970, 3, 16, 17704],
                [205970, 3, 16, 24838],
                [205970, 3, 16, 33754],
                [205970, 3, 16, 21903],
                [178520, 4, 36, 40285],
                [178520, 4, 36, 41276],
                [178520, 4, 36, 32645]],
            columns=['user_id', 'order_id',
                     'order_number', 'product_id'],
        )
        I = X.product_id.values
        gen = list(fast_bootstrap(X.values, I, 20))

    def test_user_item_ranks(self):
        np.random.seed(0)
        u, t, i, j, l = 5, 0, 0, 3, 1
        rUI = 3  # rank of UI factor
        rIL = 3  # rank of IL factor
        kU = 3
        kI = 3
        trans_probs, B = simulate(kU=kU, kI=kI,
                                  mu=0, scale=1,
                                  lamR=2, lamI=2)

        dV_ui = np.zeros(shape=(self.kU, rUI))
        dV_iu = np.zeros(shape=(self.kI, rUI))

        V_ui = np.random.normal(size=(kU, rUI))
        V_iu = np.random.normal(size=(kI, rUI))
        V_li = np.random.normal(size=(kI, rIL))
        V_il = np.random.normal(size=(kI, rIL))
        uranks = user_item_ranks(B.values, V_ui, V_iu, V_il, V_li)
        exp = np.array(
            [[0., 3.01183154, 2.84390915],
             [3.02313136, 0., 0.],
             [0., 0., 0.]]
        )
        npt.assert_allclose(exp, uranks)

    def test_update_user_matrix_overwrite(self):
        np.random.seed(0)
        u, t, i, j, l = 5, 0, 0, 3, 1
        rUI = 3  # rank of UI factor
        rIL = 3  # rank of IL factor

        dV_ui = np.zeros(shape=(self.kU, rUI))
        dV_iu = np.zeros(shape=(self.kI, rUI))

        V_ui = np.random.normal(size=(self.kU, rUI))
        V_iu = np.random.normal(size=(self.kI, rUI))
        V_li = np.random.normal(size=(self.kI, rIL))
        V_il = np.random.normal(size=(self.kI, rIL))

        exp_ui = copy.deepcopy(V_ui)
        exp_iu = copy.deepcopy(V_iu)

        update_user_matrix(
            np.array([[u, t, i, j, l]]),
            dV_iu, dV_ui,
            V_ui, V_iu,
            V_li, V_il,
            alpha=0.1, lam_ui=0, lam_iu=0)
        V_ui += dV_ui
        V_iu += dV_iu
        self.assertFalse(np.allclose(V_ui, exp_ui))
        self.assertFalse(np.allclose(V_iu, exp_iu))

    def test_update_item_matrix_overwrite(self):
        np.random.seed(0)
        u, t, i, j, l = 5, 0, 0, 3, 1
        rUI = 3  # rank of UI factor
        rIL = 3  # rank of IL factor

        dV_li = np.zeros(shape=(self.kI, rIL))
        dV_il = np.zeros(shape=(self.kI, rIL))

        V_ui = np.random.normal(size=(self.kU, rUI))
        V_iu = np.random.normal(size=(self.kI, rUI))
        V_li = np.random.normal(size=(self.kI, rIL))
        V_il = np.random.normal(size=(self.kI, rIL))

        exp_li = copy.deepcopy(V_li)
        exp_il = copy.deepcopy(V_il)
        boot = np.array([[u, t, i, j, l]], dtype=np.int64)
        update_item_matrix(
            boot, dV_il, dV_li,
            V_ui, V_iu, V_li, V_il,
            alpha=0.1, lam_il=0, lam_li=0)

        V_li += dV_li
        V_il += dV_il
        self.assertFalse(np.allclose(V_li, exp_li))
        self.assertFalse(np.allclose(V_il, exp_il))

    def test_all(self):
        np.random.seed(0)
        rUI = 3  # rank of UI factor
        rIL = 3  # rank of IL factor

        V_ui = np.random.normal(size=(self.kU, rUI))
        V_iu = np.random.normal(size=(self.kI, rUI))
        V_li = np.random.normal(size=(self.kI, rIL))
        V_il = np.random.normal(size=(self.kI, rIL))

        I = list(range(self.kI))

        uranks = user_item_ranks(self.train.values,
                                 V_ui, V_iu, V_il, V_li)
        prev_auc_score = np.mean(auc(uranks, self.test.values, I))

        for _ in range(10):
            gen = list(fast_bootstrap(self.train.values, I, 100))
            c = 0

            dV_ui = np.zeros(shape=(self.kU, rUI))
            dV_iu = np.zeros(shape=(self.kI, rUI))
            dV_li = np.zeros(shape=(self.kI, rIL))
            dV_il = np.zeros(shape=(self.kI, rIL))

            for boot in gen:
                update_user_matrix(boot, dV_ui, dV_iu,
                                   V_ui, V_iu, V_li, V_il,
                                   alpha=0.1, lam_ui=0, lam_iu=0)

                update_item_matrix(boot, dV_li, dV_il,
                                   V_ui, V_iu, V_li, V_il,
                                   alpha=0.1, lam_il=0, lam_li=0)

            V_ui += dV_ui
            V_iu += dV_iu
            V_li += dV_li
            V_il += dV_il

            for boot in gen:
                c += cost(boot,
                          V_ui, V_iu, V_li, V_il,
                          lam_ui=0, lam_iu=0,
                          lam_il=0, lam_li=0)

        # Run the metrics provided in the paper namely
        # i.e. half-life-utility
        #      precision and recall
        #      AUC under the ROC curve
        uranks = user_item_ranks(self.train.values, V_ui, V_iu, V_il, V_li)
        post_auc_score = np.mean(auc(uranks, self.test.values, I))

        self.assertGreater(post_auc_score, prev_auc_score)

        # This needs troubleshooting
        # post_prec_score, post_recall_score = score(uranks, test, I,
        #                                            method='top_precision_recall')
        # self.assertGreater(post_prec_score, prev_prec_score)
        # self.assertGreater(post_recall_score, prev_recall_score)
        # post_hlu_score = score(uranks, test, I, method='hlu')
        # self.assertGreater(post_hlu_score, prev_hlu_score)


if __name__=="__main__":
    unittest.main()
