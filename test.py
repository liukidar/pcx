import jax
import jax.numpy as jnp
jax.config.update('jax_platform_name', 'cpu')

def fit(params):
	extra_learning_rate = 0.1

	@jax.jit
	def step(params):
		print("Jitting")
		return params * extra_learning_rate

	params1 = step(params)
	print(params1)
	#extra_learning_rate = 0.01 # does this affect the next `step` call?
	params2 = step(params)
	print(params2)

	return params

x = jnp.ones(shape=(4, 2))
fit(x)