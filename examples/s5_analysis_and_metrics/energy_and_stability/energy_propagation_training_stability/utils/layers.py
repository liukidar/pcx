import math
from typing import Literal, Union

import jax.random as jrandom
from jaxtyping import PRNGKeyArray
import jax.numpy as jnp

import equinox as eqx
import pcax.nn as pxnn
from pcax.core._random import RandomKeyGenerator, RKG


class EQXLinearWithCustomInit(eqx.nn.Linear):
    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        init: str = "kaiming_uniform",
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        wkey, bkey = jrandom.split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features

        if dtype is None:
            dtype = jnp.float32

        if init == "kaiming_uniform":
            lim = math.sqrt(6) / math.sqrt(in_features_)  # assuming gain sqrt(2) 2 0.01 => 2.0201

            self.weight = jrandom.uniform(wkey, (out_features_, in_features_), minval=-lim, maxval=lim, dtype=dtype)
            if use_bias:
                self.bias = jrandom.uniform(bkey, (out_features_,), minval=-lim, maxval=lim, dtype=dtype)
            else:
                self.bias = None
        elif init == "kaiming_normal":
            std = 2 / math.sqrt(in_features_)
            self.weight = jrandom.normal(wkey, (out_features_, in_features_), dtype=dtype) * std
            if use_bias:
                self.bias = jrandom.normal(bkey, (out_features_,), dtype=dtype) * std
            else:
                self.bias = None
        elif init == "xavier_uniform":
            lim = math.sqrt(12) / math.sqrt(in_features_ + out_features_)
            self.weight = jrandom.uniform(wkey, (out_features_, in_features_), minval=-lim, maxval=lim, dtype=dtype)
            if use_bias:
                self.bias = jrandom.uniform(bkey, (out_features_,), minval=-lim, maxval=lim, dtype=dtype)
            else:
                self.bias = None
        elif init == "default":
            lim = 1 / math.sqrt(in_features_)
            self.weight = jrandom.uniform(wkey, (out_features_, in_features_), minval=-lim, maxval=lim, dtype=dtype)
            if use_bias:
                self.bias = jrandom.uniform(bkey, (out_features_,), minval=-lim, maxval=lim, dtype=dtype)
            else:
                self.bias = None
        else:
            raise ValueError(f"Unknown init method: {init}")

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias


class LinearWithCustomInit(pxnn.Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "kaiming_uniform",
        rkg: RandomKeyGenerator = RKG,
    ):
        super().__init__(EQXLinearWithCustomInit, in_features, out_features, bias, init, key=rkg())
