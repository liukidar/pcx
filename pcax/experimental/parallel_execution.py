import jax
from typing import Callable

from ..nn.layer import Linear
from ..pc.energymodule import EnergyModule, Node
from ..pc.parameters import LayerParam
from ..utils.context import vectorize
from ..core import f
from ..core.parameters import reduce_id


class pexec_MultiLinearBlock(EnergyModule):
    def __init__(self,
                 dim: int,
                 nm_layers: int,
                 act_fn: Callable[[jax.Array], jax.Array] = jax.nn.gelu):
        super().__init__()
        self.act_fn = act_fn
        self.dim = dim
        self.nm_layers = nm_layers

        # Create a single vectorized layer
        self.fc_layers = Linear(1, 1)  # dummy dimension number
        fc_layers = [Linear(dim, dim) for _ in range(nm_layers)]
        self.fc_layers.nn.weight.value = jax.numpy.stack(
            tuple(layer.nn.weight.value for layer in fc_layers),
            axis=0
        )
        self.fc_layers.nn.weight._reduce = reduce_id
        self.fc_layers.nn.bias.value = jax.numpy.stack(
            tuple(layer.nn.bias.value for layer in fc_layers),
            axis=0
        )
        self.fc_layers.nn.bias._reduce = reduce_id

        # Create a single vectorized node
        self.pc_nodes = Node()

    def __call__(self, x):
        if self.is_init:
            # Save vectorized weights and biases
            weights = self.fc_layers.nn.weight.value
            biases = self.fc_layers.nn.bias.value

            u = x

            # Uncomment the following line to simulate parallel initialization
            # (and decomment the loop below and the call to jax.numpy.stack)
            # us = jax.numpy.zeros((self.nm_layers, self.dim))

            us = []
            for i in range(self.nm_layers):
                self.fc_layers.nn.weight.value = weights[i]
                self.fc_layers.nn.bias.value = biases[i]

                u = self.act_fn(self.fc_layers(u))

                if i == self.nm_layers - 1:
                    # us = us.at[i].set(x)
                    us.append(x)
                else:
                    # us = us.at[i].set(u)
                    us.append(u)

            us = jax.numpy.stack(us, axis=0)

            # Save activations
            self.pc_nodes["x"] = us
            self.pc_nodes["us"] = us
            self.pc_nodes["u"] = x

            # Restore vectorized weights and biases
            self.fc_layers.nn.weight.value = weights
            self.fc_layers.nn.bias.value = biases

            return u
        else:
            @vectorize(f(LayerParam), in_axis=(0,), out_axis=(0,), axis_name="pexec")
            def forward(x, *, layer):
                return self.act_fn(layer(x))

            us, = forward(self.pc_nodes["x"], layer=self.fc_layers)
            self.pc_nodes(us=us, u=x)

            # Save activations
            # self.pc_nodes(jax.numpy.roll(us, shift=1, axis=0).at[-1].set(x))

            return us[-1]

    def energy(self):
        energy = 0.5 * (
            jax.numpy.square(self.pc_nodes["x"][-1] - self.pc_nodes["u"]).sum(axis=-1)
            + jax.numpy.square(self.pc_nodes["x"][:-1] - self.pc_nodes["us"][:-1]).sum(axis=-1)
        )
        return energy.sum(axis=0)

        # @vectorize(f(NodeParam, with_cache=True), in_axis=(), out_axis=("sum",), axis_name="pexec")
        # def energy(*, pc_nodes):
        #     return pc_nodes.energy()

        # return energy(pc_nodes=self.pc_nodes)
