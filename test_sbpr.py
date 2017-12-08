import unittest
import numpy as np
import numpy.testing as npt
import pandas as pd
from numpy.random import poisson, randint, random
from scipy.sparse import csr_matrix
from sbpr import (bootstrap, update_user_matrix, update_item_matrix,
                  fast_bootstrap,
                  score, simulate, split_dataset, user_item_ranks,
                  flatten, cost)
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
        trans_probs, B = simulate(kU=kU, kI=kI,
                                  mu=0, scale=1,
                                  lamR=15, lamI=5)
        self.B = B
        self.trans_probs = trans_probs
        self.kU = kU   # number of users
        self.kL = kL   # number of items
        self.kI = kI   # number of items

    def test_simulation(self):
        np.random.seed(0)
        trans_probs, B = simulate(kU=3, kI=3,
                                  mu=0, scale=1,
                                  lamR=2, lamI=2)
        exp = [[[2, 1], [1, 1], [1, 1], [0, 1, 0, 1, 0], [1, 1, 1, 1]],
               [[0, 0], [1, 0]],
               [[0, 0, 1, 1, 1, 0]]]
        self.assertListEqual(B, exp)
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

        A = pd.DataFrame(
            [[2, 202279, 3],
             [3, 205970, 16],
             [4, 178520, 36]],
            columns=['order_id', 'user_id', 'order_number'])
        A = A.set_index('order_id')
        B = pd.DataFrame(
            [[2, 33120],
             [2, 28985],
             [2, 9327 ],
             [2, 45918],
             [2, 30035],
             [2, 17794],
             [2, 40141],
             [2, 1819 ],
             [2, 43668],
             [3, 32665],
             [3, 17461],
             [3, 46667],
             [3, 17668],
             [3, 17704],
             [3, 24838],
             [3, 33754],
             [3, 21903],
             [4, 40285],
             [4, 41276],
             [4, 32645]],
            columns=['order_id', 'product_id'])
        B = B.set_index('order_id')
        I = list(set(B.product_id))
        gen = fast_bootstrap(A, B, I, 2)
        res = next(gen)
        exp = np.array([[178520, 36, 41276, 33120, 40285],
                        [178520, 36, 41276, 33120, 41276],
                        [178520, 36, 41276, 33120, 32645]])
        npt.assert_allclose(res, exp)

        res = next(gen)
        exp = np.array(
            [[205970, 16, 32665, 40285, 32665],
             [205970, 16, 32665, 40285, 17461],
             [205970, 16, 32665, 40285, 46667],
             [205970, 16, 32665, 40285, 17668],
             [205970, 16, 32665, 40285, 17704],
             [205970, 16, 32665, 40285, 24838],
             [205970, 16, 32665, 40285, 33754],
             [205970, 16, 32665, 40285, 21903]]
        )
        npt.assert_allclose(res, exp)

    def test_bootstrap(self):
        np.random.seed(0)
        I = list(range(11))
        res = bootstrap(self.B, I, 3)
        exp = np.array([[5, 1, 8, 3],
                        [5, 3, 4, 7],
                        [6, 9, 5, 1]])
        npt.assert_allclose(exp, res)

    def test_flatten(self):
        I = list(range(11))
        boots = bootstrap(self.B, I, 3)
        gen = flatten(boots, self.B)
        res = next(gen)
        exp = np.array(
            [(8, 12, 9, 0, 9),
             (8, 12, 9, 0, 1),
             (8, 12, 9, 0, 4),
             (8, 12, 9, 0, 8),
             (8, 12, 9, 0, 7),
             (8, 12, 9, 0, 9),
             (8, 12, 9, 0, 8)]
        )
        npt.assert_allclose(exp, res)

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

        I = list(range(11))

        train, test = split_dataset(self.B)
        uranks = user_item_ranks(train, V_ui, V_iu, V_il, V_li)
        prev_auc_score = score(uranks, test, I, method='auc')
        prev_prec_score, prev_recall_score = score(
            uranks, test, I,
            method='top_precision_recall')
        prev_hlu_score = score(uranks, test, I, method='hlu')

        for _ in range(10):
            boots = bootstrap(train, I, 100)
            gen = list(flatten(boots, self.B))
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
        uranks = user_item_ranks(test, V_ui, V_iu, V_il, V_li)
        post_auc_score = score(uranks, test, I, method='auc')
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
