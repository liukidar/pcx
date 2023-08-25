__all__ = [
    "cond",
    "switch",
    "scan",
]


from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from ..core.filter import f
from ..core.modules import ParamDict
from ..core.transform import _AbstractTransformation
from ..pc.energymodule import EnergyModule
from ..pc.parameters import LayerParam, NodeParam
from .context import grad_and_values, step
from .optim import Optim


class cond(_AbstractTransformation):
    def __init__(
        self,
        true_fn: Union[_AbstractTransformation, Callable],
        false_fn: Union[_AbstractTransformation, Callable],
        filter: Union[f, Callable[[ParamDict], ParamDict]] = lambda *args: True,
    ):
        super().__init__((true_fn, false_fn), filter)

    def _call(self, params_partition, *args):
        output, params_partition = self.transform(params_partition, *args)

        return output

    def _make_transform(self, fns, kwargs):
        return lambda partition, cond, *args: jax.lax.cond(
            cond,
            *tuple(self._functional(fn, kwargs) for fn in fns),
            partition,
            *args,
        )


class switch(_AbstractTransformation):
    def __init__(
        self,
        fns: Tuple[Union[_AbstractTransformation, Callable], ...],
        filter: Union[f, Callable[[ParamDict], ParamDict]] = lambda *args: True,
    ):
        super().__init__(fns, filter)

    def _call(self, params_partition, *args):
        output, params_partition = self.transform(params_partition, *args)

        return output

    def _make_transform(self, fns, kwargs):
        return lambda partition, j, *args: jax.lax.switch(
            j,
            tuple(self._functional(fn, kwargs) for fn in fns),
            partition,
            *args,
        )


class scan(_AbstractTransformation):
    def __init__(
        self,
        fn: Union[_AbstractTransformation, Callable],
        js: Optional[Union[jax.Array, Any]] = None,
        length: Optional[int] = None,
        map_outputs: Tuple[int, ...] = (),
        filter: Union[f, Callable[[ParamDict], ParamDict]] = lambda *args: True,
    ):
        assert (
            sum((js is not None, length is not None)) == 1
        ), "Exactly one between 'js' and 'length' must be specified"

        super().__init__(fn, filter)

        self.js = js
        self.length = length
        self.map_outputs = map_outputs

    def _call(self, params_partition, *args):
        if self.js is not None:
            args = (None,) + args

        (params_partition, args), output = self.transform(params_partition, *args)

        return output, *args

    def _make_transform(self, fn, kwargs):
        def scan(carry, j):
            partition, args_list = carry

            if self.js is not None:
                r, partition = self._functional(fn, kwargs)(
                    partition, j, *args_list[1:]
                )
            else:
                r, partition = self._functional(fn, kwargs)(partition, *args_list)

            # Update args
            if isinstance(r, tuple):
                if len(r) == 2 and isinstance(r[0], tuple):
                    updated_args = r[0]
                    y = r[1]
                else:
                    updated_args = r
                    y = None

                updated_args = r[0]
                for updated_arg, map_output in zip(
                    updated_args,
                    self.map_outputs
                    + tuple(range(len(updated_args) - len(self.map_outputs))),
                ):
                    args_list[map_output] = updated_arg
            else:
                y = r

            return (partition, args_list), y

        return lambda partition, *args: jax.lax.scan(
            scan, (partition, args), self.js, self.length
        )


class while_loop(_AbstractTransformation):
    def __init__(
        self,
        fn: Union[_AbstractTransformation, Callable],
        cond_fn: Callable,
        filter: Union[f, Callable[[ParamDict], ParamDict]] = lambda *args: True,
    ):
        """while_loop constructor.

        Args:
        fn: function corresponding to `body_fun` for jax.lax.while_loop,
        cond_fn: function corresponding to `cond_fun` for jax.lax.while_loop,
        filter: selects which params to apply the transformation to [
            it is used by vmap, grad, ... to select which params to be targeted by those transformations.
            There is no apparent use of it for flow transformations, but maybe I'm missing it;
            so there is still an option to specify it
        ],
        """
        super().__init__(fn, filter)

        self.cond_fn = cond_fn

    def _call(self, params_partition, *args):
        params_partition, output = self.transform(params_partition, *args)

        return output

    def _make_transform(self, fn, kwargs):
        def while_loop(carry):
            partition, args_list = carry
            updated_args, new_partition = self._functional(fn, kwargs)(
                partition, *args_list
            )
            assert len(updated_args) == len(args_list)
            self.update_partition(new_partition)

            return (self.partition, updated_args)

        return lambda partition, *args: jax.lax.while_loop(
            lambda carry: self.cond_fn(*carry[1]),
            while_loop,
            (partition, args),
        )


class EnergyMinimizationLoop:
    """Build and run a `flow.while_loop` loop for energy minimization in an EnergyModule model.

    The energy is minimized by updating X and W parameters of the model.

    The configurable loop condition function can decide to run the loop
    untill the energy has converged or the maximum number of iterations is reached.

    This is a convenience class that wraps `flow.while_loop` and takes care of all low-level details.
    """

    class LoopState(NamedTuple):
        iter_number: jax.Array
        num_x_updates_done: jax.Array
        num_w_updates_done: jax.Array

        prev_energy: jax.Array
        curr_energy: jax.Array

        all_energies: jax.Array | None

    def __init__(
        self,
        model: EnergyModule,
        loss_fn: Callable,
        max_iter_number: int,
        min_iter_number: int = 1,
        loop_condition_fn: Callable[
            [LoopState, jax.Array], jax.Array
        ] = lambda state, energy_converged: jnp.logical_not(energy_converged),
        energy_convergence_threshold: float | None = None,
        should_update_x: bool = False,
        should_update_w: bool = False,
        optim_x: Optim | None = None,
        optim_w: Optim | None = None,
        params_filter: Union[f, Callable[[ParamDict], ParamDict]] = (
            f(NodeParam)(frozen=False) | f(LayerParam)
        ),
        should_record_energies: bool = True,
    ) -> None:
        """Capture the configuration of the loop without modifying the state of the parameters.

        Args:
            model (EnergyModule): a model to be optimized.
            loss_fn (Callable): a possibly vectorized loss function that MUST return the total energy on the batch.
            max_iter_number (int): the maximum number of iterations the loop can run.
                The loop will be stopped after this number of iterations even if the energy has not converged.
            min_iter_number (int, optional): the minimum number of iterations the loop must run. Defaults to 1.
            loop_condition_fn (Callable[[LoopState, jax.Array], jax.Array], optional): a function that decides whether to continue the loop.
                The input parameters are the loop state and a scalar boolean that indicates whether the energy has converged.
                The function MUST return a scalar boolean that indicates whether to continue the loop: True -> continue.
                The energy is considered converged if the difference between energy values before and after parameters update is less than `energy_convergence_threshold`.
                The function is free to ignore the energy convergence flag.
                Note that the loop will be stopped automatically when the maximum number of iterations is reached.
                Defaults to a function that runs the loop untill the energy has converged or for the maximum number of iterations, whatever happens earlier.
            energy_convergence_threshold (float | None, optional): If not None, used to check if energy has converged. Defaults to None.
                If the difference between the loss values before and after parameters update is less than this threshold, the energy is considered converged.
            should_update_x (bool, optional): Whether the X parameters of the model should be updated using the `optim_x` optimizer. Defaults to False.
            should_update_w (bool, optional): Whether the W parameters of the model should be updated using the `optim_w` optimizer. Defaults to False.
            optim_x (Optim | None, optional): Optimizer that tracks and updates the X parameters of the model. Defaults to None.
            optim_w (Optim | None, optional): Optimizer that tracks and updates the W parameters of the model. Defaults to None.
            params_filter (Union[f, Callable[[ParamDict], ParamDict]], optional): Parameters to trace by in the grad_and_values function. Defaults to ( f(NodeParam)(frozen=False)  |  f(LayerParam) ).
            should_record_energies (bool, optional): If True, the total model energy on each iteration will be recorded in the `LoopState.all_energies` array. Defaults to True.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.loop_condition_fn = loop_condition_fn
        self.max_iter_number = max_iter_number
        if isinstance(min_iter_number, int) and min_iter_number < 1:
            raise ValueError("'min_iter_number' must be >= 1")
        self.min_iter_number = min_iter_number
        self.energy_convergence_threshold = energy_convergence_threshold
        self.should_update_x = should_update_x
        self.should_update_w = should_update_w
        if not should_update_x and not should_update_w:
            raise ValueError(
                "At least one of 'should_update_x' and 'should_update_w' must be True"
            )
        if should_update_x and optim_x is None:
            raise ValueError("'optim_x' must be specified if 'should_update_x' is True")
        if should_update_w and optim_w is None:
            raise ValueError("'optim_w' must be specified if 'should_update_w' is True")
        self.optim_x = optim_x
        self.optim_w = optim_w
        self.param_filter = params_filter
        self.should_record_energies = should_record_energies

        self.energy_submodules = list(self.model.energy_submodules())

    def _build_loop_body(self):
        _grad_and_values = grad_and_values(
            self.param_filter,  # type: ignore
        )(self.loss_fn)

        def loop_body(
            state: EnergyMinimizationLoop.LoopState,
            *loss_args,
            model: EnergyModule,
            optim_x: Optim | None = None,
            optim_w: Optim | None = None,
            **loss_kwargs,
        ) -> EnergyMinimizationLoop.LoopState:
            with step(model):
                gradients, (prev_energy,) = _grad_and_values(
                    *loss_args, model=model, **loss_kwargs
                )

                if self.should_update_x:
                    assert optim_x is not None
                    optim_x(gradients)
                if self.should_update_w:
                    assert optim_w is not None
                    optim_w(gradients)

            with step(model):
                # Re-compute energies after parameter updates
                (curr_energy,) = self.loss_fn(*loss_args, model=model, **loss_kwargs)

                if self.should_record_energies:
                    assert state.all_energies is not None
                    submodules_energies = jnp.array(
                        [jnp.sum(x.energy()) for x in self.energy_submodules]
                    )
                    all_energies = state.all_energies.at[state.iter_number].set(
                        submodules_energies
                    )
                else:
                    all_energies = None

            return (
                EnergyMinimizationLoop.LoopState(
                    iter_number=state.iter_number + 1,
                    num_x_updates_done=state.num_x_updates_done
                    + int(self.should_update_x),
                    num_w_updates_done=state.num_w_updates_done
                    + int(self.should_update_w),
                    prev_energy=jnp.sum(prev_energy),
                    curr_energy=jnp.sum(curr_energy),
                    all_energies=all_energies,
                ),
                *loss_args,
            )

        return loop_body

    def _loop_condition_fn(self, state: LoopState, *_) -> jax.Array:
        energy_converged = jnp.array(False)
        if self.energy_convergence_threshold is not None:
            energy_converged = jnp.logical_and(
                state.iter_number >= self.min_iter_number,
                jnp.abs(state.curr_energy - state.prev_energy)
                < self.energy_convergence_threshold,
            )
        return jnp.logical_and(
            state.iter_number < self.max_iter_number,
            self.loop_condition_fn(state, energy_converged),
        )

    def run(
        self,
        *loss_args,
        initial_state: LoopState | None = None,
        recording_buffer_length: int | None = None,
        **loss_kwargs,
    ) -> LoopState:
        """Build the flow.while_loop, capturing the model parameters, and run the loop untill completion.

        Use `loss_args` to pass jax.Array values to the loss function.
        Use `loss_kwargs` to pass `pcax.core.modules.Module`-derived objects to the loss function.

        The difference:
        - loss_args will be used by JAX directly to trace the computation graph of the loss function.
        - loss_kwargs will be transformed by pcax by extracting all parameters and flattening them.

        Args:
            initial_state (LoopState | None, optional): The state to start the loop from. Defaults to None.
            recording_buffer_length (int | None, optional): The length of the buffer to record energies in.
                Use when the length of the energies array should be greater than the maximum number of iterations. Defaults to None.

        Returns:
            LoopState: The final state of the loop after all iterations have been completed.
        """
        if initial_state is None:
            all_energies = None
            if self.should_record_energies:
                recording_buffer_length = (
                    recording_buffer_length or self.max_iter_number
                )
                assert recording_buffer_length >= self.max_iter_number
                all_energies = jnp.zeros(
                    (recording_buffer_length, len(self.energy_submodules)),
                    dtype=jnp.float32,
                )
            initial_state = EnergyMinimizationLoop.LoopState(
                iter_number=jnp.array(0),
                num_x_updates_done=jnp.array(0),
                num_w_updates_done=jnp.array(0),
                prev_energy=jnp.array(0.0),
                curr_energy=jnp.array(0.0),
                all_energies=all_energies,
            )

        # Make sure to remove "u" parameters from the model before passing it to the while_loop,
        # because in the loop body we use the pxu.step() decorator that calls model.clear_cache() that drops "u" parameters.
        # As the result, the list of parameters passed to the loop body and returned by the loop body differs, which is strictly forbidden by jax.lax.while_loop.
        self.model.clear_cache()

        loop = while_loop(
            self._build_loop_body(),
            self._loop_condition_fn,
        )
        outputs = loop(
            initial_state,
            *loss_args,
            model=self.model,
            optim_x=self.optim_x,
            optim_w=self.optim_w,
            **loss_kwargs,
        )
        assert len(outputs) > 0
        return outputs[0]
