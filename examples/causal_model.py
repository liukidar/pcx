# pcx
import pcx as px
import pcx.predictive_coding as pxc
import pcx.nn as pxnn
import pcx.functional as pxf
import pcx.utils as pxu

# 3rd party
import jax
from jax import jit
import jax.numpy as jnp
import jax.numpy.linalg as jax_la
import jax.random as random
import optax
import numpy as np


class Complete_Graph(pxc.EnergyModule):
    def __init__(self, input_dim: int, n_nodes: int, has_bias: bool = False, is_cont_node: list = None, seed: int = 0, stddev: float = 1.0) -> None:        
        super().__init__()

        self.input_dim = px.static(input_dim)  # Ensure input_dim is static
        self.n_nodes = px.static(n_nodes)  # Keep n_nodes as a static value
        self.has_bias = has_bias
        self.is_cont_node = is_cont_node
        self.stddev = stddev

        # Initialize a single linear layer for the weights and wrap it in a list
        self.layers = [pxnn.Linear(n_nodes, n_nodes, bias=has_bias)] # vanilla initialization is uniform(-stdv, stdv) with stdv = 1/sqrt(n_nodes), here n_nodes = 12, thus stdv = 1/sqrt(12) = 0.2887
        
        #stddev = jnp.sqrt(0.01) # this equals 0.1 (default would have been 0.2887)
        #stddev = 1/n_nodes
        stddev = self.stddev

        # Use the seed to generate a JAX key
        key = random.PRNGKey(seed)

        # split the key to ensure randomness
        #key, subkey = random.split(key) # use if more than one key is needed, ie if we do more than one random operation

        new_weight_matrix = random.normal(key, shape=(n_nodes, n_nodes)) * stddev # option 1 using normal distribution
        #new_weight_matrix = random.uniform(key, shape=(n_nodes, n_nodes), minval=-stddev, maxval=stddev) # option 2 using uniform distribution

        # Step 3: Replace diagonal elements with 0
        for i in range(n_nodes):
            new_weight_matrix = new_weight_matrix.at[i, i].set(0.0)

        # Step 5: Update the weight matrix
        self.layers[0].nn.weight.set(new_weight_matrix)

        # Initialize vodes based on is_cont_node
        if is_cont_node is None:
            is_cont_node = [True] * n_nodes  # Default to all continuous nodes if not provided

        self.vodes = []
        for is_cont in is_cont_node:
            if is_cont:
                self.vodes.append(pxc.Vode())
            else:
                #self.vodes.append(pxc.Vode(pxc.ce_energy))
                self.vodes.append(pxc.Vode(pxc.bce_energy))

    def freeze_nodes(self, freeze=True):
        """Freeze or unfreeze all vodes in the model."""
        for vode in self.vodes:
            vode.h.frozen = freeze

    def are_vodes_frozen(self):
        """Check if all vodes in the model are frozen."""
        return all(hasattr(vode.h, 'frozen') and vode.h.frozen for vode in self.vodes)
    
    def get_W(self):
        """This function returns the weighted adjacency matrix based on the linear layer in the model."""
        W = self.layers[0].nn.weight.get()
        W_T = W.T
        return W_T

    def __call__(self, x=None):
        n_nodes = self.n_nodes.get()

        if x is not None:
            # Initialize nodes with given data
            reshaped_x = x.reshape(n_nodes, -1)  # Infer input_dim from x
            
            # Print the shape of reshaped_x[0] when x is not None
            print("The shape of reshaped_x[0] when x is not None is: ", reshaped_x[0].shape)

            for i in range(n_nodes):
                self.vodes[i](reshaped_x[i])

        else:
            # Stack current state of vodes into a matrix of shape (n_nodes, input_dim)
            x_ = jnp.vstack([vode.get('h') for vode in self.vodes])

            # Print the shape of x_ when x is None
            print("The shape of x_ when x is None is: ", x_.shape)

            # Apply the linear transformation
            output = self.layers[0](x_)

            # Print the shape of output when x is None
            print("The shape of output when x is None is: ", output.shape)

            # Update the vodes with the output
            for i in range(n_nodes):
                self.vodes[i](output[i])

        # Stack the final state of vodes for output
        output = jnp.vstack([vode.get('h') for vode in self.vodes])

        # Print the shape of the output
        print("The shape of the output is: ", output.shape)

        return output
    


    ######################################### mCMIkNN #########################################

    #mcmiknn.py

### imports
import os
import csv
import logging
import multiprocessing
from multiprocessing import Pool
import networkx as nx
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from datetime import datetime
from itertools import combinations, product, chain
from math import floor
from multiprocessing import Pool, RawArray
from scipy.spatial import KDTree
from scipy.special import digamma
from scipy.stats import rankdata

### manual added implementation of mCMIkNN

class IndependenceTest(ABC):

    @abstractmethod
    def test_params(self):
        pass

    @abstractmethod
    def compute_pval(self, x, y, z):
        pass
    
class mCMIkNN(IndependenceTest):
    '''
        An independence test class that provides acces to the following non-parametric methods described in Huegle et al. (2022)
        - compute_mi: non-parametric estimator for mutual information I(X;Y)
        - compute_cmi: non-parametric estimator for conditional mutual information I(X;Y|Z)
        - compute_pval_mi: non-parametric independence test returning p value for H_0: X _||_ Y
        - compute_pval: non-parametric conditional independence test returing p value for H_0: X _||_ Y | Z
    '''
    def __init__(self,
                 kcmi=25,
                 kperm=5,
                 Mperm=100,
                 seed=42,
                 subsample=None,
                 transform=None,
                 log_warning=False):

        # Required parameters with defaults
        self.Mperm = Mperm
        self.kcmi = kcmi
        self.kperm = kperm
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)  # Initialize RNG with seed
        # Persisted values
        self.cmi_val = None
        self.null_distribution = None
        self.permutation = None
        self.pval = None
        # Options
        self.transform = transform
        self.dis = 10
        self.subsample = subsample
        self.leafsize = 16
        self.log_warning = log_warning

    def test_params(self):
        return {
            'kcmi': self.kcmi,
            'kperm':self.kperm,
            'Mperm': self.Mperm,
            'Transformation': self.transform,
        }

    def rank_transform(self, x, y, z=None):
        '''
            Rank Transform all variables while preserving rank of discrete points.
        '''
        x_transformed = rankdata(x, method='dense', axis=0).astype(np.float32)
        y_transformed = rankdata(y, method='dense', axis=0).astype(np.float32)
        z_transformed = None if np.all(z) == None else rankdata(z, method='dense', axis=0).astype(np.float32)
        return (x_transformed, y_transformed, z_transformed)

    def uniform_transform(self, x, y, z=None):
        '''
            Transform all variables to take values in [0,1] with equal distances while preserving discrete points (i.e., normalized rank transformed variables).
        '''
        x_transformed = self.normalize(rankdata(x, method='dense', axis=0).astype(np.float32))
        y_transformed = self.normalize(rankdata(y, method='dense', axis=0).astype(np.float32))
        z_transformed = None if np.all(z) == None else self.normalize(rankdata(z, method='dense', axis=0).astype(np.float32))
        return (x_transformed, y_transformed, z_transformed)

    def standardize(self, x):
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        return (x - x_mean) / x_std

    def standard_transform(self, x, y, z=None):
        '''
            Standardize all continuous variables (unique values > 10) to std. normal distribution,
            i.e., (x-mean(x))/std(x)
        '''
        res = []
        # Standardize Xa if unique(Xa) > 10
        xN = []
        for a in x.T:
            if np.unique(a, axis=0).shape[0] > self.dis:
                xN.append(self.standardize(a))
            else:
                xN.append(a)
        res.append(np.asarray(xN).T)

        # Standardize Ya if unique(Ya) > 10
        yN = []
        for a in y.T:
            if np.unique(a, axis=0).shape[0] > self.dis:
                yN.append(self.standardize(a))
            else:
                yN.append(a)
        res.append(np.asarray(yN).T)

        # Standardize Za if unique(Za) > 10
        zN = []
        if np.all(z) != None:
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
        return (x-x_min)/(x_max-x_min)

    def normal_transform(self, x, y, z=None):
        '''
            Normalize all continuous variables (unique values > 10) to take values in [0,1],
            i.e., (x-min(x))/(max(x)-min(x))
        '''
        res = []
        # Normalize Xa if unique(Xa) > 10
        xN = []
        for a in x.T:
            if np.unique(a, axis=0).shape[0] > self.dis:
                xN.append(self.normalize(a))
            else:
                xN.append(a)
        res.append(np.asarray(xN).T)

        # Normalize Ya if unique(Ya) > 10
        yN = []
        for a in y.T:
            if np.unique(a, axis=0).shape[0] > self.dis:
                yN.append(self.normalize(a))
            else:
                yN.append(a)
        res.append(np.asarray(yN).T)

        # Normalize Za if unique(Za) > 10
        zN = []
        if np.all(z) != None:
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
        '''
            Count all nearest neighbors with distance smaller or equal to rho. (Note, this does not include point itself.)
        '''
        return tree.query_ball_point(points, rho, p=np.inf, return_length=True)-1


    def return_NN(self, tree, points, sigma):
        '''
            Return all nearest neighbors with distance smaller or equal to sigma. (Note, excludes point itself.)
        '''
        # includes points itself
        neighbors = tree.query_ball_point(points, sigma, p=np.inf)
        # exclude points withing neighborhood
        for i in range(len(points)):
            neighbors[i].remove(i)

        return neighbors


    def compute_mi(self, x, y):
        '''
            Estimate the mutual information I(X;Y) of X and Y from n samples (x_i, y_i)_{i=1}^n
            using Alg. 1 which relates to the *Mixed-KSG* mutual information estimator of Gao et al. (2017)

            Note: Using digamma instead of log according to Mesner et al. (2021)

            Input:  x: 2D array of size n*dz (or 1D list of size n if dx = 1)
                    y: 2D array of size n*dz (or 1D list of size n if dy = 1)
                    (self.kcmi: k-nearest neighbor parameter)

            Output: ^I_n(X;Y)
        '''
        assert len(x) == len(y), "x and y should have the same number of observations"
        n = len(x)
        assert self.kcmi <= n-1, "Set kcmi smaller than number of observations - 1"

        x =  x.reshape((n, 1)).astype(np.float32) if (x.shape == (n,)) else x.astype(np.float32)
        y =  y.reshape((n, 1)).astype(np.float32) if (y.shape == (n,)) else y.astype(np.float32)
        xy = np.concatenate((x, y), axis=1)

        # build k-d trees
        tree_xy = KDTree(xy, leafsize=self.leafsize)
        tree_x = KDTree(x, leafsize=self.leafsize)
        tree_y = KDTree(y, leafsize=self.leafsize)

        # compute k-NN distances, using k+1 as this includes dist to self
        rho = tree_xy.query(xy, self.kcmi+1, p=np.inf)[0][:, self.kcmi]

        # if continous  -> k_tilde = k_cmi
        # if discrete or mixed -> k_tilde = number of samples with distance rho
        k_tilde = self.count_NN(tree_xy, xy, rho)

        # entropy estimates - i.e., count points withon distance rho
        nx = self.count_NN(tree_x, x, rho)
        ny = self.count_NN(tree_y, y, rho)

        mi = np.mean(digamma(k_tilde) + digamma(n) - digamma(nx) - digamma(ny))
        return max(0,mi)

    def compute_cmi(self, x, y, z):
        '''
            Estimate the conditional mutual information I(X;Y|Z) of X and Y given a dz-dimensional variable Z from samples (x_i, y_i,z_i)_{i=1}^n
            Using Alg. 1 which relates to the *Mixed-KSG* mutual information estimator of Mesner et al. (2021)

            Input:  x: 2D array of size n*dz (or 1D list of size n if dx = 1)
                    y: 2D array of size n*dz (or 1D list of size n if dy = 1)
                    z: 2D array of size n*dz (or 1D list of size n if dz = 1)
                    (self.kcmi: k-nearest neighbor parameter)

            Output: ^I_n(X;Y|Z)
        '''
        assert len(x) == len(y) == len(z), "x, y, and z should have same number of observations"
        n=len(x)
        assert self.kcmi <= n-1, "Set kcmi smaller than number of observations - 1"

        x =  x.reshape((n, 1)).astype(np.float32) if (x.shape == (n,)) else x.astype(np.float32)
        y =  y.reshape((n, 1)).astype(np.float32) if (y.shape == (n,)) else y.astype(np.float32)
        z =  z.reshape((n, 1)).astype(np.float32) if (z.shape == (n,)) else z.astype(np.float32)

        yz = np.concatenate((y, z), axis=1)
        xyz = np.concatenate((x, yz), axis=1)
        xz = np.concatenate((x, z), axis=1)

        # build k-d trees
        tree_xyz = KDTree(xyz, leafsize= self.leafsize)
        tree_xz = KDTree(xz, leafsize= self.leafsize)
        tree_yz = KDTree(yz, leafsize= self.leafsize)
        tree_z = KDTree(z, leafsize= self.leafsize)

        # compute k-NN distances, using k+1 as this includes dist to self
        rho = tree_xyz.query(xyz, self.kcmi+1, p=np.inf)[0][:, self.kcmi]

        # if continous  -> k_tilde = k_cmi
        # if discrete or mixed -> k_tilde = number of samples with distance rho
        k_tilde = self.count_NN(tree_xyz, xyz, rho)

        # entropy estimates - i.e., count neighbors within distance rho
        nxz = self.count_NN(tree_xz, xz, rho)
        nyz = self.count_NN(tree_yz, yz, rho)
        nz = self.count_NN(tree_z, z, rho)

        cmi = np.mean(digamma(k_tilde) - digamma(nxz) - digamma(nyz) + digamma(nz))
        return max(0,cmi)

    def compute_pval_mi(self, x, y):
        '''
            Returns the p value returning p value for H_0: X _||_ Y estimated from n samples (x_i, y_i)_{i=1}^n
            using Alg. 2 of Huegle et al. (2022), i.e., comparing the present MI against MIs for shuffled samples of X under H_0.

            H_0: X and Y are independent
            H_1: X and Y are dependent

            Note: Using rank transformation for k-NN searches according to Runge (2017) but preserving ties

            Input:  x: 2D array of size n*dz (or 1D list of size n if dx = 1)
                    y: 2D array of size n*dz (or 1D list of size n if dy = 1)
                    (self.Mperm: number of permutations)
                    (self.kcmi: k used for MI estimation)

            Output: p_perm,n
        '''

        assert len(x) == len(y), "x and y should have same number of observations"

        if self.subsample is not None:
            sample = np.random.choice(np.arange(len(x)), min(len(x), self.subsample), replace=False)
            x, y = x[sample], y[sample]

        n = len(x)

        x =  x.reshape((n, 1)).astype(np.float32) if (x.shape == (n,)) else x.astype(np.float32)
        y =  y.reshape((n, 1)).astype(np.float32) if (y.shape == (n,)) else y.astype(np.float32)

        x, y, _ = self.transform_data(x,y)

        n = len(x)
        if self.kperm == 0:
            self.kperm = np.floor(np.sqrt(n)).astype(int)
        elif 0 < self.kperm <= 1:
            self.kperm = np.floor(np.nextafter(self.kperm,0) * n).astype(int) # ensure kperm < n (n-1 equals shuffling without considering z)

        if self.kcmi == 0:
            self.kcmi = np.floor(np.sqrt(n)).astype(int)
        elif 0 < self.kcmi <= 1:
            self.kcmi = np.floor(np.nextafter(self.kcmi,0) * n).astype(int) # ensure kcmi < n

        # estimate present MI value
        self.cmi_val = self.compute_mi(x, y)

        # estimate Mperm MIs for shuffled X under H_0
        null_dist = np.zeros(self.Mperm)
        for m in range(self.Mperm):
             # Generate random shuffled x
            x_shuffled = x[self.rng.permutation(n)]
            null_dist[m] = self.compute_mi(x_shuffled, y)

        # estimate pvalue comparing MI against MIs for shuffled X
        self.null_distribution = null_dist
        self.pval = (1+np.sum(null_dist >= self.cmi_val))/(1+self.Mperm)
        return self.pval

    def compute_pval(self, x, y, z=None):
        '''
            Returns the p value returning p value for H_0: X _||_ Y | Z estimated from n samples (x_i, y_i,z_i)_{i=1}^n
            using Alg. 2 of Huegle et al. (2022), i.e., comparing the present MI against MIs for shuffled samples of X under H_0.

            H_0: X and Y are independent given dz dimensional Z
            H_1: X and Y are dependent given dz dimensional Z

            Note: Using rank transformation for k-NN searches according to Runge (2017) but preserving ties

            Input:  x: 1D 2D array of size n*dz (or 1D list of size n if dx = 1)
                    y: 1D 2D array of size n*dz (or 1D list of size n if dy = 1)
                    z: 2D array of size n*dz (or 1D list of size n if dz = 1)
                    (self.Mperm: number of permutations)
                    (self.kperm: k used for local permuation scheme)
                    (self.kcmi: k used for MI estimation)

            Output: p_perm,n
        '''

        # for empty z calculate return p value according to H_0: X _||_ Y
        if z is None:
            return self.compute_pval_mi(x, y)

        assert len(x) == len(y) == len(z), "x, y, and z should have same number of observations"

        if self.subsample is not None:
            sample = np.random.choice(np.arange(len(x)), min(len(x), self.subsample), replace=False)
            x, y, z = x[sample], y[sample], z[sample]

        n = len(x)

        x =  x.reshape((n, 1)).astype(np.float32) if (x.shape == (n,)) else x.astype(np.float32)
        y =  y.reshape((n, 1)).astype(np.float32) if (y.shape == (n,)) else y.astype(np.float32)
        z =  z.reshape((n, 1)).astype(np.float32) if (z.shape == (n,)) else z.astype(np.float32)
        x, y, z = self.transform_data(x, y, z)

        if self.kperm == 0:
            self.kperm = np.floor(np.sqrt(n)).astype(int)
        elif 0 < self.kperm <= 1:
            self.kperm = np.floor(np.nextafter(self.kperm,0) * n).astype(int) # ensure kperm < n (n-1 equals shuffling without considering z)

        if self.kcmi == 0:
            self.kcmi = np.floor(np.sqrt(n)).astype(int)
        elif 0 < self.kcmi <= 1:
            self.kcmi = np.floor(np.nextafter(self.kcmi,0) * n).astype(int) # ensure kcmi < n

        # estimate present CMI value
        self.cmi_val = self.compute_cmi(x, y, z)

        # Get nearest neighbors around each sample point in Z
        tree_z = KDTree(z, leafsize= self.leafsize)

        # compute k-NN distances in Z, using k+1 as this includes dist to self
        sigma = tree_z.query(z, self.kperm+1, p=np.inf)[0][:, self.kperm]

        # if continuous -> k points distance smaller or equal to sigma excluding the point itself
        # if discrete or mixed -> all points with distance smaller or euqla to the k-NN distance sigma excluding the point itself
        neighbors = self.return_NN(tree_z, z, sigma)

        # estimate Mperm CMIs for shuffled X under H_0 while preserving marginal distributions
        null_dist = np.zeros(self.Mperm)
        for m in range(self.Mperm):
            # compose local permutations of nearest neighbors to receive a restricted permutation of the whole index list
            permutation = np.arange(n)
            for i in range(n-1,-1,-1):
                permutation[neighbors[i,]]=permutation[self.rng.permutation(neighbors[i,])]
            x_shuffled = x[permutation]
            null_dist[m] = self.compute_cmi(x_shuffled, y, z)

        self.null_distribution = null_dist
        self.pval = (1+np.sum(null_dist >= self.cmi_val))/(1+self.Mperm)
        return self.pval
