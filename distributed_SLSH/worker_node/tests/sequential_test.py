import unittest
from worker_node.SLSH.hash_family import *
from worker_node.SLSH.slsh import *
from worker_node.SLSH.selectors import *
from worker_node.query import Query


class TestHashFamily(unittest.TestCase):
    def test_L1(self):
        H = L1LSH([2, 3, 4])
        f = H.sample_fn()
        val = f([1, 1.5, 2])
        self.assertTrue(val in [0, 1])

    def test_COS(self):
        H = COSLSH(2)
        f = H.sample_fn()
        points = [(0, 1), (1, 0), (0.7071, 0.7071)]
        for point in points:
            val = f(point)
            self.assertTrue(val in [0, 1])


class TestLSH(unittest.TestCase):
    def test_L1_NN(self):
        # Test the output is the correct NN.
        D = 50
        H = L1LSH([5] * D)
        X1_matrix = np.eye(D)  # The dataset is a unit matrix of size D.
        X1 = np.reshape(np.transpose(X1_matrix), D * D).tolist()
        X1_shape = (D, D)

        m = 10
        L = 10
        k = 1

        points = range(D)  # Hash all the points.
        (g, T1) = lsh_indexing(m, L, X1, X1_shape, points, H)

        x = np.array(X1[10 * X1_shape[0]:11 * X1_shape[0]])
        query = Query(x * 1.0001)

        selector = NearestPoints(X=X1, X_shape=X1_shape)
        nn = lsh_querying(query, T1, g, k, selector)

        self.assertTrue(np.array_equal(X1_matrix[nn[0]], x))


class TestSLSH(unittest.TestCase):
    def test_NN(self):
        # Test a query's NN corresponds to what it should be.
        D = 50
        H_out = L1LSH([5] * D)
        H_in = COSLSH(D)

        X_matrix = np.eye(D)  # The dataset is a unit matrix of size D.
        X = np.reshape(np.transpose(X_matrix), D * D).tolist()
        X_shape = (D, D)

        m_out = 10
        L_out = 10
        m_in = 10
        L_in = 5
        k = 1
        alpha = 0.2

        (T_out, g_out, T_in, g_in) = slsh_indexing(m_out, L_out, m_in, L_in, X,
                                                   X_shape, H_out, H_in, alpha)

        x = np.array(X[20 * X_shape[0]:21 * X_shape[0]])
        query = Query(x * 2)

        selector = NearestPoints(X=X, X_shape=X_shape)
        nn = slsh_querying(query, T_out, g_out, T_in, g_in, k, selector)

        self.assertTrue(np.array_equal(X_matrix[nn[0]], x))


if __name__ == '__main__':
    unittest.main()
