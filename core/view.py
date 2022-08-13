from typing import Callable, List, Any, Type, Tuple
import jax.numpy as jnp

class View:
	name: str
	parent: None|'View'|jnp.array
	children: List['View']
	transformation_fn: None|Callable
	cached: None|jnp.array
	boundaries: Any # TODO

	def __init__(self, _name:str = None, _transformation_fn:Callable = None, _children:List['View'] = None, _boundaries = None) -> None:
		self.name = _name
		self.parent = None
		self.children = _children if _children is not None else []
		self.transformation_fn = _transformation_fn
		self.cached = None

		if _boundaries is not None:
			raise NotImplementedError("Boundaries for a View are not currently supported.")

		for child in self.children:
			child.parent = self

	def get(self, _apply_t:bool = False, _cache:bool = False) -> jnp.array:
		if _cache == True and self.cached is not None:
			return self.cached
		
		if issubclass(self.parent, View):
			x = self.parent.get()
		elif isinstance(self.parent, jnp.array):
			x = self.parent
		else:
			raise ValueError("Parent cannot be None when querying a View.")

		if _apply_t:
			x = self.transformation_fn(x)

		if _cache == True:
			self.cached = x

		return x

	def set(self, _x:jnp.array, _cache:bool = False):		
		if issubclass(self.parent, View):
			self.parent.set(_x, _cache)
		elif isinstance(self.parent, jnp.array):
			self.parent = _x
		else:
			raise ValueError("Parent cannot be None when querying a View.")

		if _cache == True:
			self.cached = self.transformation_fn(_x)
	
	def flush_cache(self):
		self.cached = None

		for child in self.children:
			child.flush_cache()

	def find(self, _path:str) -> 'View':
		_path = _path.split('/', 1)
		target = None

		if _path[0] == '.':
			target = self
		elif _path[0] == '..':
			assert issubclass(self.parent, View)
			target = self.parent
		else:
			for child in self.children:
				if child.name == _path[0]:
					target = child
					break
		
		if len(_path) == 1:
			return target
		else:
			return target.find(_path[1])

	def clone(self, _target:str = None, _name:str = None, _type:Type['View'] = None, **kwargs):
		if _target is None:
			_target = self
		elif _target == '..':
			assert issubclass(self.parent, View)
			_target = self.parent
			
			if _name is None:
				raise NotImplementedError("A valid name needs to be specified")
		elif _target == '.':
			assert issubclass(self.parent, View)
			self.parent.children.remove(self) 
			_target = self.parent

		if _type is None:
			_type = type(self)

		if _name is None:
			_name = self._name

		new_view = _type(_name, **kwargs)
		new_view.parent = self

		_target.append(new_view)

	def split(indices:Any, target:str = None, type:Type['View']|Tuple[Type['View']] = None, 
		name:str|Tuple[str] = None, *args, **kwargs):
		raise NotImplementedError()

class OutputView(View):
	energy_fn: None|Callable

	def __init__(self, _name:str = None, _transformation_fn:Callable = None, _children:Tuple['View'] = None, _boundaries = None, _energy_fn:Callable = None) -> None:
		super().__init__(_name, _transformation_fn, _children, _boundaries)
		self.energy_fn = _energy_fn

	def set(self, _x:jnp.array):		
		self.cached = self.transformation_fn(_x)

	def get(self, _cat_mode:str='cat', _cache:bool = True, **_kwargs) -> jnp.array|Tuple[jnp.array]:
		if _cache == True and self.cached is not None:
			return self.cached

		r = list((child.get(_cat_mode, _cache, **_kwargs) for child in self.children))

		if _cat_mode == 'cat':
			args = {
				'axis': _kwargs.get('axis', -1)
			}
			r = jnp.concatenate(r, **args)

		if _cache == True:
			self.cached = r

		return r

	def clone(self, _target:str = None, _name:str = None, **_kwargs):
		return super(View, self).clone(_target, _name, type=OutputView, **_kwargs)

class InputView(View):
	def get(self, _cache:bool = True) -> jnp.array:
		return super(View, self).get(_apply_t=True, _cache=_cache)

	def clone(self, _target:str = None, _name:str = None, **_kwargs):
		return super(View, self).clone(_target, _name, type=InputView, **_kwargs)