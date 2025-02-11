# pc_parallel.py

import os
import logging
import multiprocessing
from multiprocessing import Pool, RawArray
import networkx as nx
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from datetime import datetime
from itertools import combinations, product
from math import floor
from scipy.spatial import KDTree
from scipy.special import digamma
from scipy.stats import rankdata

# Define the abstract independence test interface
class IndependenceTest(ABC):
    @abstractmethod
    def test_params(self):
        pass

    @abstractmethod
    def compute_pval(self, x, y, z):
        pass

# Implementation of mCMIkNN
class mCMIkNN(IndependenceTest):
    def __init__(self, kcmi=25, kperm=5, Mperm=100, subsample=None, transform=None, log_warning=False):
        self.Mperm = Mperm
        self.kcmi = kcmi
        self.kperm = kperm
        self.cmi_val = None
        self.null_distribution = None
        self.permutation = None
        self.pval = None
        self.transform = transform
        self.dis = 10
        self.subsample = subsample
        self.leafsize = 16
        self.log_warning = log_warning

    def test_params(self):
        return {
            'kcmi': self.kcmi,
            'kperm': self.kperm,
            'Mperm': self.Mperm,
            'Transformation': self.transform,
        }

    def rank_transform(self, x, y, z=None):
        x_transformed = rankdata(x, method='dense', axis=0).astype(np.float32)
        y_transformed = rankdata(y, method='dense', axis=0).astype(np.float32)
        z_transformed = None if np.all(z) is None else rankdata(z, method='dense', axis=0).astype(np.float32)
        return (x_transformed, y_transformed, z_transformed)

    def uniform_transform(self, x, y, z=None):
        x_transformed = self.normalize(rankdata(x, method='dense', axis=0).astype(np.float32))
        y_transformed = self.normalize(rankdata(y, method='dense', axis=0).astype(np.float32))
        z_transformed = None if np.all(z) is None else self.normalize(rankdata(z, method='dense', axis=0).astype(np.float32))
        return (x_transformed, y_transformed, z_transformed)

    def standardize(self, x):
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        return (x - x_mean) / x_std

    def standard_transform(self, x, y, z=None):
        res = []
        xN = []
        for a in x.T:
            if np.unique(a, axis=0).shape[0] > self.dis:
                xN.append(self.standardize(a))
            else:
                xN.append(a)
        res.append(np.asarray(xN).T)

        yN = []
        for a in y.T:
            if np.unique(a, axis=0).shape[0] > self.dis:
                yN.append(self.standardize(a))
            else:
                yN.append(a)
        res.append(np.asarray(yN).T)

        zN = []
        if np.all(z) is not None:
            for a in z.T:
                if np.unique(a, axis=0).shape[0] > self.dis:
                    zN.append(self.standardize(a))
                else:
                    zN.append(a)
            res.append(np.asarray(zN).T)
        else:
            res.append(None)
        return res

    def normalize(self, x):
        x_min = np.min(x, axis=0)
        x_max = np.max(x, axis=0)
        return (x - x_min) / (x_max - x_min)

    def normal_transform(self, x, y, z=None):
        res = []
        xN = []
        for a in x.T:
            if np.unique(a, axis=0).shape[0] > self.dis:
                xN.append(self.normalize(a))
            else:
                xN.append(a)
        res.append(np.asarray(xN).T)

        yN = []
        for a in y.T:
            if np.unique(a, axis=0).shape[0] > self.dis:
                yN.append(self.normalize(a))
            else:
                yN.append(a)
        res.append(np.asarray(yN).T)

        zN = []
        if np.all(z) is not None:
            for a in z.T:
                if np.unique(a, axis=0).shape[0] > self.dis:
                    zN.append(self.normalize(a))
                else:
                    zN.append(a)
            res.append(np.asarray(zN).T)
        else:
            res.append(None)
        return res

    def transform_data(self, x, y, z=None):
        if self.transform == 'rank':
            return self.rank_transform(x, y, z)
        elif self.transform == 'standardize':
            return self.standard_transform(x, y, z)
        elif self.transform == 'normalize':
            return self.normal_transform(x, y, z)
        elif self.transform == 'uniform':
            return self.uniform_transform(x, y, z)
        return (x, y, z)

    def count_NN(self, tree, points, rho):
        return tree.query_ball_point(points, rho, p=np.inf, return_length=True) - 1

    def return_NN(self, tree, points, sigma):
        neighbors = tree.query_ball_point(points, sigma, p=np.inf)
        for i in range(len(points)):
            neighbors[i].remove(i)
        return neighbors

    def compute_mi(self, x, y):
        assert len(x) == len(y), "x and y should have the same number of observations"
        n = len(x)
        assert self.kcmi <= n - 1, "Set kcmi smaller than number of observations - 1"
        x = x.reshape((n, 1)).astype(np.float32) if (x.shape == (n,)) else x.astype(np.float32)
        y = y.reshape((n, 1)).astype(np.float32) if (y.shape == (n,)) else y.astype(np.float32)
        xy = np.concatenate((x, y), axis=1)
        tree_xy = KDTree(xy, leafsize=self.leafsize)
        tree_x = KDTree(x, leafsize=self.leafsize)
        tree_y = KDTree(y, leafsize=self.leafsize)
        rho = tree_xy.query(xy, self.kcmi + 1, p=np.inf)[0][:, self.kcmi]
        k_tilde = self.count_NN(tree_xy, xy, rho)
        nx_val = self.count_NN(tree_x, x, rho)
        ny_val = self.count_NN(tree_y, y, rho)
        mi = np.mean(digamma(k_tilde) + digamma(n) - digamma(nx_val) - digamma(ny_val))
        return max(0, mi)

    def compute_cmi(self, x, y, z):
        assert len(x) == len(y) == len(z), "x, y, and z should have same number of observations"
        n = len(x)
        assert self.kcmi <= n - 1, "Set kcmi smaller than number of observations - 1"
        x = x.reshape((n, 1)).astype(np.float32) if (x.shape == (n,)) else x.astype(np.float32)
        y = y.reshape((n, 1)).astype(np.float32) if (y.shape == (n,)) else y.astype(np.float32)
        z = z.reshape((n, 1)).astype(np.float32) if (z.shape == (n,)) else z.astype(np.float32)
        yz = np.concatenate((y, z), axis=1)
        xyz = np.concatenate((x, yz), axis=1)
        xz = np.concatenate((x, z), axis=1)
        tree_xyz = KDTree(xyz, leafsize=self.leafsize)
        tree_xz = KDTree(xz, leafsize=self.leafsize)
        tree_yz = KDTree(yz, leafsize=self.leafsize)
        tree_z = KDTree(z, leafsize=self.leafsize)
        rho = tree_xyz.query(xyz, self.kcmi + 1, p=np.inf)[0][:, self.kcmi]
        k_tilde = self.count_NN(tree_xyz, xyz, rho)
        nxz = self.count_NN(tree_xz, xz, rho)
        nyz = self.count_NN(tree_yz, yz, rho)
        nz = self.count_NN(tree_z, z, rho)
        cmi = np.mean(digamma(k_tilde) - digamma(nxz) - digamma(nyz) + digamma(nz))
        return max(0, cmi)

    def compute_pval_mi(self, x, y):
        assert len(x) == len(y), "x and y should have same number of observations"
        if self.subsample is not None:
            sample = np.random.choice(np.arange(len(x)), min(len(x), self.subsample), replace=False)
            x, y = x[sample], y[sample]
        n = len(x)
        x = x.reshape((n, 1)).astype(np.float32) if (x.shape == (n,)) else x.astype(np.float32)
        y = y.reshape((n, 1)).astype(np.float32) if (y.shape == (n,)) else y.astype(np.float32)
        x, y, _ = self.transform_data(x, y)
        n = len(x)
        if self.kperm == 0:
            self.kperm = np.floor(np.sqrt(n)).astype(int)
        elif 0 < self.kperm <= 1:
            self.kperm = np.floor(np.nextafter(self.kperm, 0) * n).astype(int)
        if self.kcmi == 0:
            self.kcmi = np.floor(np.sqrt(n)).astype(int)
        elif 0 < self.kcmi <= 1:
            self.kcmi = np.floor(np.nextafter(self.kcmi, 0) * n).astype(int)
        self.cmi_val = self.compute_mi(x, y)
        null_dist = np.zeros(self.Mperm)
        for m in range(self.Mperm):
            x_shuffled = x[np.random.default_rng().permutation(n)]
            null_dist[m] = self.compute_mi(x_shuffled, y)
        self.null_distribution = null_dist
        self.pval = (1 + np.sum(null_dist >= self.cmi_val)) / (1 + self.Mperm)
        return self.pval

    def compute_pval(self, x, y, z=None):
        if z is None:
            return self.compute_pval_mi(x, y)
        assert len(x) == len(y) == len(z), "x, y, and z should have same number of observations"
        if self.subsample is not None:
            sample = np.random.choice(np.arange(len(x)), min(len(x), self.subsample), replace=False)
            x, y, z = x[sample], y[sample], z[sample]
        n = len(x)
        x = x.reshape((n, 1)).astype(np.float32) if (x.shape == (n,)) else x.astype(np.float32)
        y = y.reshape((n, 1)).astype(np.float32) if (y.shape == (n,)) else y.astype(np.float32)
        z = z.reshape((n, 1)).astype(np.float32) if (z.shape == (n,)) else z.astype(np.float32)
        x, y, z = self.transform_data(x, y, z)
        if self.kperm == 0:
            self.kperm = np.floor(np.sqrt(n)).astype(int)
        elif 0 < self.kperm <= 1:
            self.kperm = np.floor(np.nextafter(self.kperm, 0) * n).astype(int)
        if self.kcmi == 0:
            self.kcmi = np.floor(np.sqrt(n)).astype(int)
        elif 0 < self.kcmi <= 1:
            self.kcmi = np.floor(np.nextafter(self.kcmi, 0) * n).astype(int)
        self.cmi_val = self.compute_cmi(x, y, z)
        tree_z = KDTree(z, leafsize=self.leafsize)
        sigma = tree_z.query(z, self.kperm + 1, p=np.inf)[0][:, self.kperm]
        neighbors = self.return_NN(tree_z, z, sigma)
        null_dist = np.zeros(self.Mperm)
        for m in range(self.Mperm):
            permutation = np.arange(n)
            for i in range(n - 1, -1, -1):
                permutation[neighbors[i]] = permutation[np.random.default_rng().permutation(neighbors[i])]
            x_shuffled = x[permutation]
            null_dist[m] = self.compute_cmi(x_shuffled, y, z)
        self.null_distribution = null_dist
        self.pval = (1 + np.sum(null_dist >= self.cmi_val)) / (1 + self.Mperm)
        return self.pval

# Merge the PC algorithm into a single class
class PCParallel:
    # Class variable to share data with worker processes.
    var_dict = {}

    def __init__(self, alpha=0.05, processes=2, kcmi=25, kperm=5, Mperm=100, max_level=None):
        self.alpha = alpha
        self.processes = processes
        self.kcmi = kcmi
        self.kperm = kperm
        self.Mperm = Mperm
        self.max_level = max_level
        self.estimator = mCMIkNN(kcmi=self.kcmi, kperm=self.kperm, Mperm=self.Mperm)

    @staticmethod
    def _init_worker(data, data_shape, graph, vertices, test, alpha):
        import os
        # Tell the worker process to run JAX on CPU
        os.environ["JAX_PLATFORMS"] = "cpu"
        # Optionally, hide GPU from castle/torch as well:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["CASTLE_BACKEND"] = ""
        PCParallel.var_dict = {}
        PCParallel.var_dict['data'] = data
        PCParallel.var_dict['data_shape'] = data_shape
        PCParallel.var_dict['graph'] = graph
        PCParallel.var_dict['vertices'] = vertices
        PCParallel.var_dict['alpha'] = alpha
        PCParallel.var_dict['test'] = test

    @staticmethod
    def _test_worker(i, j, lvl):
        test = PCParallel.var_dict['test']
        alpha = PCParallel.var_dict['alpha']
        data_arr = np.frombuffer(PCParallel.var_dict['data']).reshape(PCParallel.var_dict['data_shape'])
        graph = np.frombuffer(PCParallel.var_dict['graph'], dtype="int32").reshape(
            (PCParallel.var_dict['vertices'], PCParallel.var_dict['vertices']))
        if lvl < 1:
            p_val = test.compute_pval(data_arr[:, [i]], data_arr[:, [j]], z=None)
            if p_val > alpha:
                return (i, j, p_val, [])
        else:
            candidates_1 = np.arange(PCParallel.var_dict['vertices'])[graph[i] == 1]
            candidates_1 = np.delete(candidates_1, np.argwhere((candidates_1 == i) | (candidates_1 == j)))
            if len(candidates_1) < lvl:
                return None
            for S in [list(c) for c in combinations(candidates_1, lvl)]:
                p_val = test.compute_pval(data_arr[:, [i]], data_arr[:, [j]], z=data_arr[:, list(S)])
                if p_val > alpha:
                    return (i, j, p_val, list(S))
        return None

    @staticmethod
    def _unid(g, i, j):
        return g.has_edge(i, j) and not g.has_edge(j, i)

    @staticmethod
    def _bid(g, i, j):
        return g.has_edge(i, j) and g.has_edge(j, i)

    @staticmethod
    def _adj(g, i, j):
        return g.has_edge(i, j) or g.has_edge(j, i)

    @staticmethod
    def rule1(g, j, k):
        for i in g.predecessors(j):
            if PCParallel._unid(g, i, j) and not PCParallel._adj(g, i, k):
                g.remove_edge(k, j)
                return True
        return False

    @staticmethod
    def rule2(g, i, j):
        for k in g.successors(i):
            if PCParallel._unid(g, k, j) and PCParallel._unid(g, i, k):
                g.remove_edge(j, i)
                return True
        return False

    @staticmethod
    def rule3(g, i, j):
        for k, l in combinations(g.predecessors(j), 2):
            if (not PCParallel._adj(g, k, l) and PCParallel._bid(g, i, k) and
                    PCParallel._bid(g, i, l) and PCParallel._unid(g, l, j) and PCParallel._unid(g, k, j)):
                g.remove_edge(j, i)
                return True
        return False

    @staticmethod
    def rule4(g, i, j):
        for l in g.predecessors(j):
            for k in g.predecessors(l):
                if (not PCParallel._adj(g, k, j) and PCParallel._adj(g, i, l) and
                        PCParallel._unid(g, k, l) and PCParallel._unid(g, l, j) and PCParallel._bid(g, i, k)):
                    g.remove_edge(j, i)
                    return True
        return False

    @staticmethod
    def _direct_edges(graph, sepsets):
        digraph = nx.DiGraph(graph)
        for i in graph.nodes():
            for j in nx.non_neighbors(graph, i):
                for k in nx.common_neighbors(graph, i, j):
                    sepset = sepsets[(i, j)] if (i, j) in sepsets else []
                    if k not in sepset:
                        if (k, i) in digraph.edges() and (i, k) in digraph.edges():
                            digraph.remove_edge(k, i)
                        if (k, j) in digraph.edges() and (j, k) in digraph.edges():
                            digraph.remove_edge(k, j)
        bidirectional_edges = [(i, j) for i, j in digraph.edges() if digraph.has_edge(j, i)]
        for i, j in bidirectional_edges:
            if PCParallel._bid(digraph, i, j):
                continue
            if (PCParallel.rule1(digraph, i, j) or PCParallel.rule2(digraph, i, j) or
                    PCParallel.rule3(digraph, i, j) or PCParallel.rule4(digraph, i, j)):
                continue
        return digraph

    def parallel_stable_pc(self, data, estimator, alpha, processes, max_level):
        cols = data.columns
        cols_map = np.arange(len(cols))
        data_raw = RawArray('d', data.shape[0] * data.shape[1])
        data_arr = np.frombuffer(data_raw).reshape(data.shape)
        np.copyto(data_arr, data.values)
        vertices = len(cols)
        graph_raw = RawArray('i', np.ones(vertices * vertices).astype(int))
        graph = np.frombuffer(graph_raw, dtype="int32").reshape((vertices, vertices))
        sepsets = {}
        lvls = range((len(cols) - 1) if max_level is None else min(len(cols) - 1, max_level + 1))
        for lvl in lvls:
            configs = [(i, j, lvl) for i, j in product(cols_map, cols_map)
                       if i != j and graph[i][j] == 1]
            logging.info(f'Starting level {lvl} pool with {len(configs)} remaining edges at {datetime.now()}')
            with Pool(processes=processes, initializer=PCParallel._init_worker,
                      initargs=(data_raw, data.shape, graph_raw, vertices, estimator, alpha)) as pool:
                result = pool.starmap(PCParallel._test_worker, configs)
            for r in result:
                if r is not None:
                    graph[r[0]][r[1]] = 0
                    graph[r[1]][r[0]] = 0
                    sepsets[(r[0], r[1])] = {'p_val': r[2], 'sepset': r[3]}
        nx_graph = nx.from_numpy_array(graph)
        nx_graph.remove_edges_from(nx.selfloop_edges(nx_graph))
        nx_digraph = PCParallel._direct_edges(nx_graph, sepsets)
        nx.relabel_nodes(nx_digraph, lambda i: cols[i], copy=False)
        sepsets = {(cols[k[0]], cols[k[1]]): {'p_val': v['p_val'],
                                                'sepset': [cols[e] for e in v['sepset']]}
                   for k, v in sepsets.items()}
        return nx_digraph, sepsets

    def run(self, df):
        graph, sepsets = self.parallel_stable_pc(df, self.estimator, self.alpha,
                                                   self.processes, self.max_level)
        return graph, sepsets

    def run_from_file(self, filename, counter=0, data_path='../data_generation/csl_data_normalized/'):
        print(counter, 'processing file', filename)
        df = pd.read_csv(os.path.join(data_path, filename))
        return self.run(df)
