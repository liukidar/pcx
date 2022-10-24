from typing import List
from .view import View, InputView, OutputView
import itertools
import jax.numpy as jnp


class EnergyCriterion:
    def __init__(self, cache_values: bool = False) -> None:
        self.cache_values = cache_values
        self._state = []

    def __call__(self, energy_view: View) -> jnp.ndarray:
        root = energy_view[0]
        view = energy_view[1]
        input_views = tuple(filter(lambda el: isinstance(el, InputView), view.children))
        output_views = tuple(
            filter(
                lambda el: isinstance(el, OutputView) and el.energy_fn is not None,
                view.children,
            )
        )

        energy_pairs = tuple(
            filter(
                lambda el: self.is_compatible(*el),
                itertools.product(input_views, output_views),
            )
        )

        energies = jnp.stack(
            list(map(lambda ep: ep[1].energy_fn(root, ep[0], ep[1]), energy_pairs)),
            axis=0,
        )

        if self.cache_values:
            self._state.append((energy_view, energies))
        return jnp.sum(energies, axis=0)

    def is_compatible(self, input_view: InputView, output_view: OutputView) -> bool:
        #TODO
        return True

    def clear_cache(self) -> None:
        self._state = []

    def pop_cache(self) -> List:
        state = self._state.pop()
        self.clear_cache()
        return state


def gaussian_energy(mu, x):
    t = mu - x

    return 0.5 * jnp.dot(t, t)
