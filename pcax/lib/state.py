from typing import Any, Callable, Dict, List, Tuple
import equinox as eqx
import jax.tree_util as jtu
from pcax.core.node import NodeInfo
from pcax.lib.tree import tree_mask
from pcax.utils.functions import ensure_list


class Param:
    def __init__(self, data) -> None:
        if isinstance(data, Param):
            self.data = data.data
        else:
            self.data = data

    def get(self):
        return self.data

    def set(self, value):
        self.data = value


def is_param(x: Any, state: Any = None):
    return isinstance(x, Param) or eqx.is_array(x)


class FrozenParam(Param):
    def set(self, _):
        raise NotImplementedError("You cannot assign a value to a frozen parameter.")


class UnfrozenParam(Param):
    pass


def __unfrozen_param_set_attr(self, name, value):
    assert name == "data"

    return object.__setattr__(self, name, value)


UnfrozenParam.__setattr__ = __unfrozen_param_set_attr


def _freeze_param(param):
    object.__setattr__(param, "__class__", FrozenParam)

    return param


def _unfreeze_param(param):
    object.__setattr__(param, "__class__", UnfrozenParam)

    return param


def _freeze_model(model):
    def replace_fn(param):
        return param.get() if isinstance(param, Param) else param

    return jtu.tree_map(lambda param: replace_fn(param), model)


def _unfreeze_model(model, filter_fn=None, filter_args: List[Any | str] = []):
    def replace_fn(param, *args):
        if filter_fn is None or filter_fn(param, *args):
            # make sure that everything is a Param before unfreezing
            return _unfreeze_param(Param(param))
        else:
            return param

    return jtu.tree_map(
        replace_fn,
        model,
        *filter_args,
    )


class _State(eqx.Module):
    static_masks: Dict[str, Any] = eqx.static_field()
    dynamic_masks: Dict[str, Any]
    actions: List[Callable] = eqx.static_field()

    class _FreezeManager:
        def __init__(self, model, filter_fn=None, filter_args=None) -> None:
            self._model = model
            self._filter_fn = filter_fn
            self._filter_args = filter_args
            self._unfrozen = False

        def __enter__(self):
            self._model = _unfreeze_model(
                self._model, self._filter_fn, self._filter_args
            )
            self._unfrozen = True

            return self

        def __exit__(self, exc_type, *_):
            assert self._unfrozen is False, (
                "You need to freeze the model by calling"
                " 'state.freeze(unforzen_model)'"
                " before exiting the 'with' statement block"
            )
            if exc_type is not None:

                return False

            return True

        def __getattr__(self, __name: str) -> Any:
            return getattr(self._model, __name)

        def __call__(self) -> Any:
            return self._model

    class _Itemgetter:
        def __init__(self, dict):
            self.dict = dict

        def __getitem__(self, keys: str | Tuple[str]) -> Any:
            if isinstance(keys, str):
                return self.dict[keys]
            else:
                return [self.dict[key] for key in keys]

        def __call__(self) -> Any:
            return self.dict.values()

    def __init__(self, actions: List["StateAction"] = []) -> None:
        self.static_masks = {}
        self.dynamic_masks = {}
        self.actions = {action._name: action for action in actions}

    def __hash__(self):
        return hash(frozenset(self.static_masks.items()))

    def init(
        self,
        model: Any,
        action_names: List[str],
        actions_kwargs: Dict[str, Any] = {},
        **kwargs
    ):
        return self.exec_actions(model, action_names, actions_kwargs, **kwargs)

    def exec_actions(
        self,
        model: Any,
        action_names: List[str],
        actions_kwargs: Dict[str, Any] = {},
        **kwargs
    ):
        if action_names == "*":
            # We rely on the fact that from python 3.7
            # the order of the keys in a dict is guaranteed to be the insertion order
            action_names = self.actions.keys()

        main_kwargs = {"state": self, "model": model}
        for action in action_names:
            r_kwargs = self.actions[action]._exec(
                **main_kwargs, **{**actions_kwargs.get(action, {}), **kwargs}
            )
            main_kwargs.update(r_kwargs)

        return _State._Itemgetter(main_kwargs)

    def create_mask(
        self,
        pytree: Any,
        mask_fn: Callable[[Any], Any],
        mask_name: str = None,
        mask_args: List[Any | str] = None,
        root: Any = None,
        root_name: str = None,
        is_leaf=None,
    ) -> Any:
        assert (
            root is None or mask_args is None
        ), "You cannot specify both 'root' and 'use_maps'"

        if root is not None:
            mask = tree_mask(
                pytree, mask_fn, root_state=root, is_leaf=is_leaf, name=root_name
            )
        else:
            mask = jtu.tree_map(
                pytree,
                mask_fn,
                *self.get_masks(mask_args),
                is_leaf=is_leaf,
            )

        if mask_name is not None:
            self.save_mask(mask_name, mask)

        return mask

    def map_mask(
        self,
        map_fn: Callable[[Any], Any],
        map_args: List[Any | str],
        is_leaf: Callable[[Any], bool] = None,
        mask_name: str = None,
    ) -> None:
        mask = jtu.tree_map(
            lambda *masks: map_fn(*masks),
            *self.get_masks(map_args),
            is_leaf=is_leaf,
        )

        if mask_name is not None:
            self.save_mask(mask_name, mask)

        return mask

    def save_mask(self, name: str, mask: Any, type: str = "static") -> None:
        if type == "static":
            self.static_masks[name] = mask
        elif type == "dynamic":
            self.dynamic_masks[name] = mask

    # 'names' is a list of the names of the masks to get,
    # if an element is not a string, it is instead passed through.
    # This is useful for passing arguments to filter/map functions
    # in a compact way.
    def get_masks(self, names: List[Any | str]) -> List[Any]:
        names = ensure_list(names)

        return list(
            (
                (
                    self.static_masks
                    if name in self.static_masks
                    else self.dynamic_masks
                )[name]
                if isinstance(name, str)
                else name
            )
            for name in (names or ())
        )

    def partition(
        self,
        model: Any,
        filter_fn: Callable[[NodeInfo], bool],
        filter_args: List[Any | str],
        is_leaf: Callable[[Any], bool] = None,
    ) -> Tuple[Any, Any]:
        return eqx.partition(
            model,
            jtu.tree_map(
                lambda model, *masks: filter_fn(model, *masks),
                model,
                *self.get_masks(filter_args),
                is_leaf=is_leaf,
            ),
        )

    def freeze(self, unfrozen_model):
        unfrozen_model._unfrozen = False

        return _freeze_model(unfrozen_model())

    def unfreeze(
        self,
        frozen_model,
        filter_fn: Callable[[Any], bool] = lambda _, type: type is not None,
        filter_args: List[Any | str] = "type",
    ):
        return _State._FreezeManager(
            frozen_model, filter_fn, self.get_masks(filter_args)
        )


class StateAction:
    def __init__(self, _name, **kwargs):
        self.kwargs = kwargs
        self._name = _name

    def _exec(self, state, model, **kwargs):
        return self(state, model, **{**self.kwargs, **kwargs})
