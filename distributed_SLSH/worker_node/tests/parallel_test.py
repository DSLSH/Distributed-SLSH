import unittest
import numpy as np
from worker_node.SLSH.hash_family import *
from worker_node.SLSH.lsh import *
from worker_node.SLSH.slsh import *
from worker_node.SLSH.selectors import *
from worker_node.node import *
from worker_node.query import Query


class TestParallelSLSH(unittest.TestCase):
    def test_parallel_NN(self):
        # Test a query's NN corresponds to what it should be, on 2 cores.
        cores = 3
        D = 80
        H_out = L1LSH([(-1, 1)] * D)
        H_in = COSLSH(D)

        X = np.eye(D)  # The dataset is a unit matrix of size D.

        m_out = 50
        L_out = 50
        m_in = 20
        L_in = 10
        k = 1
        alpha = 0.01

        # Create query and expected result.
        x = X[21]
        query = Query(x * 2)

        # Execute parallel code.
        temp1, queries = execute_node(
            cores,
            k,
            m_out,
            L_out,
            m_in,
            L_in,
            H_out,
            H_in,
            alpha,
            X=X,
            queries=[query])

        print("In: {}".format(x * 2))
        print("Out: {}".format(queries[0].neighbors[0]))

        self.assertTrue(np.array_equal(queries[0].neighbors[0], x))
