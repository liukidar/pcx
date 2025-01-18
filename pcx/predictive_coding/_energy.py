__all__ = ["zero_energy", "se_energy", "ce_energy", "bce_energy"]


import jax

from ..core._random import RKG, RandomKeyGenerator


########################################################################################################################
#
# Energy
#
# Collection of the most common energy functions used in predictive coding.
#
########################################################################################################################


# Core #################################################################################################################


def zero_energy(vode, rkg: RandomKeyGenerator = RKG):
    """used to unconstrain the value of a vode from its prior distribution (i.e., input)."""
    return jax.numpy.zeros((1,))


def se_energy(vode, rkg: RandomKeyGenerator = RKG):
    """Squared error energy function derived from a Gaussian distribution."""
    e = vode.get("h") - vode.get("u")
    return 0.5 * (e * e)


def ce_energy(vode, rkg: RandomKeyGenerator = RKG):
    """Cross entropy energy function derived from a categorical distribution."""
    return -(vode.get("h") * jax.nn.log_softmax(vode.get("u")))


def bce_energy(vode, rkg: RandomKeyGenerator = RKG):
    """Binary cross entropy energy function derived from a Bernoulli distribution."""
    h = vode.get("h")  # Observed binary variable (0 or 1)
    u = vode.get("u")  # Logits
    return -(h * jax.nn.log_sigmoid(u) + (1 - h) * jax.nn.log_sigmoid(-u))