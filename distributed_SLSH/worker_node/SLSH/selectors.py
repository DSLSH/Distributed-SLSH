import collections
from abc import ABCMeta
import numpy as np
from sklearn.neighbors import NearestNeighbors


class Selector:
    """
    Class that defines a data structure where items are accumulated,
    and how the resulting k things are filtered
    """

    __metaclass__ = ABCMeta

    def add(self, items):
        """
        :param item: list of points to add to data structure
        :type item: list of tuples
        :returns: None
        """
        pass

    def filter(self, d, k, q):
        """
        :param d: distance function
        :type d: function
        :param k: # of points
        :type k: int
        :param q: query point
        :type q: tuple
        :returns: list of k closest points to q
        """
        pass

    def candidates(self):
        """
        :returns: list of candidates in data structure
        """
        return list(self.data)


class NearestPoints(Selector):
    """
    Selector that returns k nearest points
    """

    def __init__(self, X=[], X_shape=[], use_dataset=True, prediction=False):
        '''
        This Selector works both with explicit points and with Point ID's.
        In the latter case, the shared dataset X and its shape X_shape are necessary to compute distances.
        Intranode it is ideal to use ID's, in the middleware points.

        :param X: shared dataset
        :param X_shape: dataset shape
        :param use_dataset: flag True if the point ID's + shared dataset are used.
        :param prediction: flag True if selection is performed for prediction (useful only if use_dataset is False).
        '''
        self.data = []
        self.X = X
        self.X_shape = X_shape
        self.use_dataset = use_dataset
        self.prediction = prediction

    def add(self, items):
        # Complexity: O(1)

        self.data.append(
            items
        )  # Add whole lists to move the linear time to the filter function.

    def filter(self, d, k, q, save_comparisons=False):

        # Complexity: O(number of elements in data).

        if not self.prediction:

            # This is faster than numpy's duplicate removal for matrices, as it compares hashes of vectors (it is, though, approximate).
            complete_list = iterate_and_remove_duplicates(self.data)
            if save_comparisons:
                comparisons = len(complete_list)
                q.comparisons = comparisons

            # Node usage.
            if self.use_dataset:
                # Assumes we're using L1 norm.
                return self.filter_faster(complete_list, d, k, q.point)

            dists = np.empty(len(complete_list))
            points = np.empty((len(complete_list[0]), len(complete_list)))
            for i in range(len(complete_list)):
                element = complete_list[i]
                x = element  # i is already a point in this usage
                points[:, i] = element

                dists[i] = d(x, q.point)

            # Account for the possibility of having less than k candidates.
            found_points = len(dists)
            if found_points < k:
                k = found_points

            selected_indices = np.argsort(dists)[:k]
            return [points[:, index] for index in selected_indices]

        else:
            return self.filter_labeled(d, k, q.point)

    def filter_labeled(self, d, k, q):

        # Same behavior of filter, but works only with points (not point ID's) and labels.
        assert not self.use_dataset

        complete_list = iterate_and_remove_duplicates(
            self.data, prediction=True)

        if len(complete_list) == 0:
            return [], []

        dists = []
        for e in complete_list:
            dists.append((d(e[0], q), e))

        dists = np.empty(len(complete_list))
        points = np.empty((len(complete_list[0][0]), len(complete_list)))
        labels = np.empty(len(complete_list))
        for i in range(len(complete_list)):
            element = complete_list[i]
            x = element[0]
            points[:, i] = x
            dists[i] = d(x, q)
            labels[i] = element[1]

        # Account for the possibility of having less than k candidates.
        found_points = len(dists)
        if found_points < k:
            k = found_points

        selected_indices = np.argsort(dists)[:k]
        filtered_points = [points[:, index] for index in selected_indices]
        filtered_labels = labels[selected_indices].tolist()

        return filtered_points, filtered_labels

    def filter_faster(self, complete_list, d, k, q):

        if d(2 * np.ones(3), np.zeros(3)) != 6:
            raise ValueError("Unimplemented norm for filter_faster")

        if len(complete_list) == 0:
            return []
        elif len(complete_list) < k:
            k = len(complete_list)

        points = np.empty((len(complete_list), self.X_shape[0]))
        counter = 0
        for i in complete_list:
            # i is a point index
            points[counter, :] = self.X[i * self.X_shape[0]:(
                i + 1) * self.X_shape[0]]
            counter += 1

        scikitmodel = NearestNeighbors(
            n_neighbors=k, algorithm='brute', metric='manhattan')
        scikitmodel.fit(points)
        knn_results = scikitmodel.kneighbors(
            X=q.reshape(1, -1), return_distance=False)[0].astype(int)
        local_query_result = np.array(complete_list)[knn_results]

        return local_query_result.tolist()


class MostFrequent(Selector):
    """
    Selector that returns k most frequent points in list
    """

    def __init__(self):
        self.data = collections.Counter()

    def add(self, items):
        self.data.update(items)

    def filter(self, d, k, q):
        return self.data.most_common(k)


class NoFilter(Selector):
    """
    Just collect a list of elements
    """

    def __init__(self):
        self.data = []

    def add(self, items):
        self.data.extend(items)

    def filter(self, d, k, q):
        return self.data


def iterate_and_remove_duplicates(data, prediction=False):
    """
    Given data (list of lists) containing n elements, it returns the list itself without duplicates in O(n) time.
    A set is used for the purpose.
    It works both with ID's and points.

    :param data: list of lists containing the input.
    :param prediction: if True, the elements of the list are a (point, label) pair.
    :return: the list without duplicates
    """

    # Use a dictionary as set to cope with numpy arrays.
    cset = dict()

    #TODO: possible optimization. Use numpy to remove duplicates when they're indices.

    # If the input is a list of numpy arrays, add a hashable version of them.
    if not prediction:
        for clist in data:
            for e in clist:
                if isinstance(e, np.ndarray):
                    cset[hash(e.data.tobytes())] = e
                elif isinstance(e, list):
                    cset[hash(tuple(e))] = e
                else:
                    cset[e] = e
    else:
        # Prediction case, on middleware.
        point_list = []
        label_list = []
        for clist in data:
            point_list += clist[0]
            label_list += clist[1]
        complete_list = []
        for i in range(len(point_list)):
            complete_list.append((point_list[i], label_list[i]))

        for e in complete_list:
            if isinstance(e[0], np.ndarray):
                cset[hash(e[0].data.tobytes())] = e
            elif isinstance(e[0], list):
                cset[hash(tuple(e[0]))] = e
            else:
                cset[e[0]] = e

    return list(cset.values())
